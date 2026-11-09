// Adapted from FerrisMind/candle-video ebe43c414d2519535357113cad22d15fba0ce900.
// Copyright 2025 FerrisMind. Apache-2.0; see reference/LICENSE.
// Aster modifications: token-wise temporal modulation for reference conditioning.
use candle_core::{IndexOp, Result, Tensor};
use candle_nn as nn;
use candle_video::models::ltx_video::ltx_transformer::LtxVideoTransformerBlock as TextBlock;
use candle_video::models::ltx_video::ltx_transformer::{
    AdaLayerNormSingle, FeedForward, LayerNormNoParams, LtxAttention, LtxVideoRotaryPosEmbed,
    LtxVideoTransformer3DModelConfig, PixArtAlphaTextProjection, RmsNorm,
};
use candle_video::models::ltx_video::t2v_pipeline::{TransformerConfig, VideoTransformer3D};
use nn::{Module, VarBuilder};
#[derive(Clone, Debug)]
pub struct LtxVideoTransformerBlock {
    norm1: RmsNorm,
    attn1: super::portrait_attention::SelfAttention,
    norm2: RmsNorm,
    attn2: LtxAttention,
    ff: FeedForward,
    scale_shift_table: Tensor, // [6, dim]
}

impl LtxVideoTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        num_attention_heads: usize,
        attention_head_dim: usize,
        cross_attention_dim: usize,
        qk_norm: &str,
        attention_bias: bool,
        attention_out_bias: bool,
        eps: f64,
        elementwise_affine: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = RmsNorm::new(dim, eps, elementwise_affine, vb.pp("norm1"))?;
        let attn1 = super::portrait_attention::SelfAttention::new(
            dim,
            num_attention_heads,
            attention_head_dim,
            attention_bias,
            attention_out_bias,
            vb.pp("attn1"),
        )?;
        let norm2 = RmsNorm::new(dim, eps, elementwise_affine, vb.pp("norm2"))?;
        let attn2 = LtxAttention::new(
            dim,
            num_attention_heads,
            num_attention_heads,
            attention_head_dim,
            0.0,
            attention_bias,
            Some(cross_attention_dim),
            attention_out_bias,
            qk_norm,
            vb.pp("attn2"),
        )?;

        let ff = FeedForward::new(dim, vb.pp("ff"))?;

        // Parameter: torch.randn(6, dim) / dim**0.5
        // In candle: we store as a trainable tensor; initialization is delegated to checkpoint loading.
        let scale_shift_table = vb.get((6, dim), "scale_shift_table")?;

        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            ff,
            scale_shift_table,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,         // [B, S, dim]
        encoder_hidden_states: &Tensor, // [B, K, dim]
        temb: &Tensor,                  // [B, T, 6*dim]
        image_rotary_emb: Option<(&Tensor, &Tensor)>,
        encoder_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let b = hidden_states.dim(0)?;
        let norm_hidden = self.norm1.forward(hidden_states)?;

        // ada_values = scale_shift_table[None,None] + temb.reshape(B, T, 6, dim)
        // ada_values = scale_shift_table[None,None] + temb.reshape(B, T, 6, dim)
        let (b_temb, t, temb_last) = temb.dims3()?;
        if b_temb != b {
            candle_core::bail!(
                "temb batch size {} mismatch hidden_states batch size {}",
                b_temb,
                b
            );
        }

        if temb_last % 6 != 0 {
            candle_core::bail!("temb last dim must be divisible by 6, got {temb_last}");
        }
        let dim = temb_last / 6;
        if t != 1 && t != hidden_states.dim(1)? {
            candle_core::bail!("time embedding must have one value or one per token");
        }
        let temb_reshaped = temb.reshape((b, t, 6, dim))?;

        let table = self
            .scale_shift_table
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((b, t, 6, dim))?;
        let ada = table.broadcast_add(&temb_reshaped)?; // [B,T,6,dim]

        let shift_msa = ada.i((.., .., 0, ..))?;
        let scale_msa = ada.i((.., .., 1, ..))?;
        let gate_msa = ada.i((.., .., 2, ..))?;
        let shift_mlp = ada.i((.., .., 3, ..))?;
        let scale_mlp = ada.i((.., .., 4, ..))?;
        let gate_mlp = ada.i((.., .., 5, ..))?;

        // norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        // Align shapes: norm_hidden is [B,S,dim], while shift/scale are [B,T,dim].
        // In the Python file, T corresponds to the second dimension after time embedding; typically T==1.
        // We broadcast T over S when T==1.

        let norm_hidden = {
            let one = Tensor::ones_like(&scale_msa)?;
            let s = one.broadcast_add(&scale_msa)?; // 1+scale
            // If T==1, expand to [B,S,dim]
            let s = if s.dim(1)? == 1 {
                s.broadcast_as((b, hidden_states.dim(1)?, s.dim(2)?))?
            } else {
                s
            };
            let sh = if shift_msa.dim(1)? == 1 {
                shift_msa.broadcast_as((b, hidden_states.dim(1)?, shift_msa.dim(2)?))?
            } else {
                shift_msa
            };
            norm_hidden.broadcast_mul(&s)?.broadcast_add(&sh)?
        };

        // Self-attn (encoder_hidden_states=None) with RoPE
        let attn1 = self.attn1.forward(&norm_hidden, image_rotary_emb)?;
        let gate_msa = if gate_msa.dim(1)? == 1 {
            gate_msa.broadcast_as((b, hidden_states.dim(1)?, gate_msa.dim(2)?))?
        } else {
            gate_msa
        };
        let mut hs = hidden_states.broadcast_add(&attn1.broadcast_mul(&gate_msa)?)?;

        // Cross-attn
        let attn2 = self.attn2.forward(
            &hs,
            Some(encoder_hidden_states),
            encoder_attention_mask,
            None,
        )?;
        hs = hs.broadcast_add(&attn2)?;

        // MLP
        let norm2 = self.norm2.forward(&hs)?;
        let norm2 = {
            let one = Tensor::ones_like(&scale_mlp)?;
            let s = one.broadcast_add(&scale_mlp)?;
            let s = if s.dim(1)? == 1 {
                s.broadcast_as((b, hs.dim(1)?, s.dim(2)?))?
            } else {
                s
            };
            let sh = if shift_mlp.dim(1)? == 1 {
                shift_mlp.broadcast_as((b, hs.dim(1)?, shift_mlp.dim(2)?))?
            } else {
                shift_mlp
            };
            norm2.broadcast_mul(&s)?.broadcast_add(&sh)?
        };
        let ff = self.ff.forward(&norm2)?;
        let gate_mlp = if gate_mlp.dim(1)? == 1 {
            gate_mlp.broadcast_as((b, hs.dim(1)?, gate_mlp.dim(2)?))?
        } else {
            gate_mlp
        };
        hs = hs.broadcast_add(&ff.broadcast_mul(&gate_mlp)?)?;

        Ok(hs)
    }
}

pub struct LtxVideoTransformer3DModel {
    proj_in: nn::Linear,
    scale_shift_table: Tensor, // [2, inner_dim]
    time_embed: AdaLayerNormSingle,
    caption_projection: PixArtAlphaTextProjection,
    rope: LtxVideoRotaryPosEmbed,
    transformer_blocks: Vec<LtxVideoTransformerBlock>,
    streamed: Option<(
        crate::prefetch::Stream,
        LtxVideoTransformer3DModelConfig,
        bool,
    )>,
    norm_out: LayerNormNoParams,

    proj_out: nn::Linear,
    pipeline_config: TransformerConfig,
    skip_block_list: Vec<usize>,
}

impl LtxVideoTransformer3DModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(config: &LtxVideoTransformer3DModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::build(config, vb, None)
    }

    pub fn from_disk(
        config: &LtxVideoTransformer3DModelConfig,
        path: &std::path::Path,
        device: &candle_core::Device,
        text_only: bool,
    ) -> crate::weights::Result<Self> {
        let file = crate::prefetch::Stream::open(path, device, config.num_layers)?;
        let vb = file.shared(device)?;
        Ok(Self::build(
            config,
            vb,
            Some((file, config.clone(), text_only)),
        )?)
    }

    fn build(
        config: &LtxVideoTransformer3DModelConfig,
        vb: VarBuilder,
        streamed: Option<(
            crate::prefetch::Stream,
            LtxVideoTransformer3DModelConfig,
            bool,
        )>,
    ) -> Result<Self> {
        let out_channels = if config.out_channels == 0 {
            config.in_channels
        } else {
            config.out_channels
        };
        let inner_dim = config.num_attention_heads * config.attention_head_dim;

        let proj_in = nn::linear(config.in_channels, inner_dim, vb.pp("proj_in"))?;

        let scale_shift_table = vb.get((2, inner_dim), "scale_shift_table")?;

        let time_embed = AdaLayerNormSingle::new(inner_dim, vb.pp("time_embed"))?;
        let caption_projection = PixArtAlphaTextProjection::new(
            config.caption_channels,
            inner_dim,
            vb.pp("caption_projection"),
        )?;

        let rope = LtxVideoRotaryPosEmbed::new(
            inner_dim,
            20,
            2048,
            2048,
            config.patch_size,
            config.patch_size_t,
            10000.0,
        );

        let mut transformer_blocks = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..if streamed.is_none() {
            config.num_layers
        } else {
            0
        } {
            transformer_blocks.push(LtxVideoTransformerBlock::new(
                inner_dim,
                config.num_attention_heads,
                config.attention_head_dim,
                config.cross_attention_dim,
                &config.qk_norm,
                config.attention_bias,
                config.attention_out_bias,
                config.norm_eps,
                config.norm_elementwise_affine,
                vb.pp("transformer_blocks").pp(layer_idx.to_string()),
            )?);
        }

        let norm_out = LayerNormNoParams::new(1e-6);
        let proj_out = nn::linear(inner_dim, out_channels, vb.pp("proj_out"))?;

        Ok(Self {
            proj_in,
            scale_shift_table,
            time_embed,
            caption_projection,
            rope,
            transformer_blocks,
            streamed,
            norm_out,
            proj_out,
            pipeline_config: TransformerConfig {
                in_channels: config.in_channels,
                patch_size: config.patch_size,
                patch_size_t: config.patch_size_t,
                num_layers: config.num_layers,
            },
            skip_block_list: Vec::new(),
        })
    }

    pub fn set_skip_block_list(&mut self, list: Vec<usize>) {
        self.skip_block_list = list;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,                  // [B, S, in_channels]
        encoder_hidden_states: &Tensor,          // [B, K, caption_channels]
        timestep: &Tensor,                       // int tensor, shape [B] or [B,1]...
        encoder_attention_mask: Option<&Tensor>, // [B,K] (1 valid, 0 pad) or already bias
        num_frames: usize,
        height: usize,
        width: usize,
        rope_interpolation_scale: Option<(f64, f64, f64)>,
        video_coords: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, sequence, _) = hidden_states.dims3()?;
        let time_tokens = match timestep.dims() {
            [b] if *b == batch => 1,
            [b, t] if *b == batch && (*t == 1 || *t == sequence) => *t,
            _ => candle_core::bail!("timestep must be [batch], [batch,1] or [batch,sequence]"),
        };

        // Convert inputs to model dtype (BF16) if needed
        let model_dtype = self.proj_in.weight().dtype();
        let hidden_states = hidden_states.to_dtype(model_dtype)?;
        let encoder_hidden_states = encoder_hidden_states.to_dtype(model_dtype)?;

        let hidden_states = self.proj_in.forward(&hidden_states)?;

        let timestep = timestep.flatten_all()?.to_dtype(model_dtype)?; // ensure [B] or flat, in BF16

        // 1. AdaLayerNormSingle (Timesteps -> PixArtEmbed -> Silu -> Linear)
        let (temb, embedded_timestep) = self.time_embed.forward(&timestep)?; // [B, 6*dim], [B, dim]
        let temb = temb.reshape((batch, time_tokens, temb.dim(1)?))?;
        let embedded_timestep =
            embedded_timestep.reshape((batch, time_tokens, embedded_timestep.dim(1)?))?;

        let encoder_hidden_states = self.caption_projection.forward(&encoder_hidden_states)?;

        // convert encoder_attention_mask to a bias
        let encoder_attention_mask = if let Some(mask) = encoder_attention_mask {
            if mask.rank() == 2 {
                let mask_f = mask.to_dtype(hidden_states.dtype())?;
                // (1 - mask) * -10000.0
                let bias = (mask_f.affine(-1.0, 1.0)? * (-10000.0))?;
                Some(bias.unsqueeze(1)?)
            } else {
                Some(mask.clone())
            }
        } else {
            None
        };
        let encoder_attention_mask = encoder_attention_mask.as_ref();

        let (cos, sin) = self.rope.forward(
            &hidden_states,
            num_frames,
            height,
            width,
            rope_interpolation_scale,
            video_coords,
        )?;

        // Pass embedded_timestep as conditioning? LTX-Video adds it to temb or similar?
        // Wait, logic in `forward` block of transformer_ltx.py:
        // temb is passed to blocks. embedded_timestep is not used explicitly in blocks for this model version?
        // Let's check block forward. Block uses `temb`.
        // The `AdaLayerNormSingle` returns (temb, embedded_timestep), but `LtxVideoTransformerBlock`
        // seems to take `temb`.
        // `LtxVideoTransformerBlock::forward` signature: `temb: &Tensor`.

        let mut hidden_states = hidden_states;
        let image_rotary_emb = Some((&cos, &sin));

        let reader = self
            .streamed
            .as_ref()
            .map(|(stream, _, _)| {
                stream.start(self.pipeline_config.num_layers, &self.skip_block_list)
            })
            .transpose()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            .flatten();
        for index in 0..self.pipeline_config.num_layers {
            if self.skip_block_list.contains(&index) {
                continue;
            }

            let original_hidden_states = if skip_layer_mask.is_some() {
                Some(hidden_states.clone())
            } else {
                None
            };

            hidden_states = if let Some((file, c, text_only)) = &self.streamed {
                let device = hidden_states.device();
                let vb = file
                    .block(index, device, reader.as_ref())
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                    .pp("transformer_blocks")
                    .pp(index.to_string());
                // Keep each backend's original attention and conditioning operations.
                if *text_only {
                    let block = TextBlock::new(
                        c.num_attention_heads * c.attention_head_dim,
                        c.num_attention_heads,
                        c.attention_head_dim,
                        c.cross_attention_dim,
                        &c.qk_norm,
                        c.attention_bias,
                        c.attention_out_bias,
                        c.norm_eps,
                        c.norm_elementwise_affine,
                        vb,
                    )?;
                    let y = block.forward(
                        &hidden_states,
                        &encoder_hidden_states,
                        &temb.squeeze(1)?,
                        image_rotary_emb,
                        encoder_attention_mask,
                    )?;
                    device.synchronize()?;
                    y
                } else {
                    let block = LtxVideoTransformerBlock::new(
                        c.num_attention_heads * c.attention_head_dim,
                        c.num_attention_heads,
                        c.attention_head_dim,
                        c.cross_attention_dim,
                        &c.qk_norm,
                        c.attention_bias,
                        c.attention_out_bias,
                        c.norm_eps,
                        c.norm_elementwise_affine,
                        vb,
                    )?;
                    let y = block.forward(
                        &hidden_states,
                        &encoder_hidden_states,
                        &temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                    )?;
                    device.synchronize()?;
                    y
                }
            } else {
                self.transformer_blocks[index].forward(
                    &hidden_states,
                    &encoder_hidden_states,
                    &temb,
                    image_rotary_emb,
                    encoder_attention_mask,
                )?
            };

            if let (Some(mask), Some(orig)) = (skip_layer_mask, original_hidden_states) {
                // mask shape: [num_layers, batch]
                // FIX: mask=1 means SKIP layer (use original), mask=0 means APPLY layer (keep processed)
                let m = mask.narrow(0, index, 1)?.flatten_all()?;
                let b_size = hidden_states.dim(0)?;
                let m = m.reshape((b_size, 1, 1))?.to_dtype(hidden_states.dtype())?;
                let one_minus_m = m.affine(-1.0, 1.0)?;
                // When m=1 (skip): use orig. When m=0 (apply): use hidden_states
                hidden_states = hidden_states
                    .broadcast_mul(&one_minus_m)? // m=0 -> keep processed hidden_states
                    .broadcast_add(&orig.broadcast_mul(&m)?)?; // m=1 -> add original (skip)
            }
        }

        // Final modulation: scale_shift_table[None,None] + embedded_timestep[:, :, None]
        let b = hidden_states.dim(0)?;
        let inner_dim = hidden_states.dim(2)?;

        // scale_shift_table: [2, inner_dim] -> cast to dtype of embedded_timestep
        let table = self.scale_shift_table.to_dtype(embedded_timestep.dtype())?;

        // table: [1, 1, 2, inner_dim]
        let table = table.unsqueeze(0)?.unsqueeze(0)?;

        // embedded_timestep: [B, T=1, inner_dim] (usually T=1 after pool or similar? AdaLayerNormSingle returns [B, inner_dim]?)
        // AdaLayerNormSingle returns `emb` which is [B, inner_dim].
        // We need to check dims.
        // If embedded_timestep is [B, D].
        let emb = embedded_timestep.unsqueeze(2)?; // [B,T,1,D]

        // broadcast add: table + emb
        // [1,1,2,D] + [B,1,1,D] -> [B,1,2,D]
        let scale_shift = table.broadcast_add(&emb)?;

        let shift = scale_shift.i((.., .., 0, ..))?; // [B, 1, D]
        let scale = scale_shift.i((.., .., 1, ..))?; // [B, 1, D]

        let mut hidden_states = self.norm_out.forward(&hidden_states)?;

        // (1 + scale) * x + shift
        let one = Tensor::ones_like(&scale)?;
        let ss = one.broadcast_add(&scale)?;

        // Broadcast scale/shift to [B, S, D]
        // S is dim 1 of hidden_states
        let s_dim = hidden_states.dim(1)?;
        let ss = ss.broadcast_as((b, s_dim, inner_dim))?;
        let sh = shift.broadcast_as((b, s_dim, inner_dim))?;

        hidden_states = hidden_states.broadcast_mul(&ss)?.broadcast_add(&sh)?;

        let hidden_states = self.proj_out.forward(&hidden_states)?;

        // Residual connection? In Python:
        // output = self.proj_out(self.norm_out(hidden_states))
        // return output + residual (if configured? No usually strict functional)
        // Check if `hidden_states` (input) is added?
        // LTX models usually treat it as noise prediction (epsilon or v-prediction).
        // Let's assume just return output.
        Ok(hidden_states)
    }
}

impl VideoTransformer3D for LtxVideoTransformer3DModel {
    fn config(&self) -> &TransformerConfig {
        &self.pipeline_config
    }

    fn set_skip_block_list(&mut self, list: Vec<usize>) {
        self.set_skip_block_list(list);
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        timestep: &Tensor,
        encoder_attention_mask: &Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
        rope_interpolation_scale: Option<(f32, f32, f32)>,
        video_coords: Option<&Tensor>,
        skip_layer_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // cast scale to f64
        let scale = rope_interpolation_scale.map(|s| (s.0 as f64, s.1 as f64, s.2 as f64));

        // Call inherent forward
        // Note: inherent forward takes &self, trait takes &mut self (which coerces to &self).
        LtxVideoTransformer3DModel::forward(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            Some(encoder_attention_mask),
            num_frames,
            height,
            width,
            scale,
            video_coords,
            skip_layer_mask,
        )
    }
}
