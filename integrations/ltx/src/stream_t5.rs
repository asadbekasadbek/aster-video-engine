// Adapted from FerrisMind/candle-video ebe43c414d2519535357113cad22d15fba0ce900.
// Apache-2.0, copyright FerrisMind. See reference/LICENSE.
// Use the original T5 operations while fetching one GGUF block at a time.
use candle_core::{
    DType, Device, Module, Result, Shape, Tensor,
    quantized::{QTensor, gguf_file},
};
use candle_video::models::ltx_video::quantized_t5_encoder::T5EncoderConfig;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    sync::Arc,
};

struct Reader {
    file: File,
    index: gguf_file::Content,
}

// GGUF stores complete quantization blocks consecutively within each vocabulary
// row. Gather the encoded rows first, then use Candle's original dequantizer.
fn prompt_embeddings(
    reader: &mut (impl Read + Seek),
    start: u64,
    info: &gguf_file::TensorInfo,
    ids: &Tensor,
) -> Result<Tensor> {
    let (vocabulary, width) = info.shape.dims2()?;
    let format = info.ggml_dtype;
    if !width.is_multiple_of(format.block_size()) || width == 0 {
        candle_core::bail!("GGUF token rows must contain whole quantization blocks");
    }
    let row_bytes = (width / format.block_size())
        .checked_mul(format.type_size())
        .ok_or_else(|| candle_core::Error::Msg("GGUF token row too large".into()))?;
    let tokens = ids.flatten_all()?.to_vec1::<u32>()?;
    if tokens.is_empty() || tokens.iter().any(|&id| id as usize >= vocabulary) {
        candle_core::bail!("invalid T5 token IDs");
    }
    let mut unique = tokens.clone();
    unique.sort_unstable();
    unique.dedup();
    let mut packed = vec![
        0u8;
        unique.len().checked_mul(row_bytes).ok_or_else(|| {
            candle_core::Error::Msg("GGUF token buffer too large".into())
        })?
    ];
    let base = start
        .checked_add(info.offset)
        .ok_or_else(|| candle_core::Error::Msg("invalid token table offset".into()))?;
    for (&id, dst) in unique.iter().zip(packed.chunks_exact_mut(row_bytes)) {
        let offset = (id as u64)
            .checked_mul(row_bytes as u64)
            .and_then(|v| base.checked_add(v))
            .ok_or_else(|| candle_core::Error::Msg("invalid token row offset".into()))?;
        reader.seek(SeekFrom::Start(offset))?;
        reader.read_exact(dst)?;
    }
    let weights = candle_core::quantized::ggml_file::qtensor_from_ggml(
        format,
        &packed,
        vec![unique.len(), width],
        &Device::Cpu,
    )?
    .dequantize(&Device::Cpu)?;
    let local = tokens
        .iter()
        .map(|id| {
            unique
                .binary_search(id)
                .expect("unique contains all input IDs") as u32
        })
        .collect::<Vec<_>>();
    let indices = Tensor::from_vec(local, ids.shape(), &Device::Cpu)?;
    candle_nn::Embedding::new(weights, width).forward(&indices)
}
impl Reader {
    fn get(&mut self, shape: impl Into<Shape>, name: &str) -> Result<Arc<QTensor>> {
        let expected = shape.into();
        let info = self
            .index
            .tensor_infos
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing {name}")))?;
        if info.shape != expected {
            candle_core::bail!("invalid GGUF shape for {name}");
        }
        Ok(Arc::new(self.index.tensor(
            &mut self.file,
            name,
            &Device::Cpu,
        )?))
    }
}

fn gelu_new(x: &Tensor) -> Result<Tensor> {
    let x_cube = (x.sqr()? * x)?;
    let inner = (x + (x_cube * 0.044715)?)?;
    let inner = (inner * (2.0f64 / std::f64::consts::PI).sqrt())?;
    let tanh = inner.tanh()?;
    (x * 0.5f64)?.broadcast_mul(&(tanh + 1.0)?)
}

/// Quantized linear layer using QTensor
struct QLinear {
    weight: Arc<QTensor>,
}

impl QLinear {
    fn new(weight: Arc<QTensor>) -> Self {
        Self { weight }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = self.weight.dequantize(x.device())?;
        let in_dims = x.dims();
        let batch_dims = &in_dims[..in_dims.len() - 1];
        let in_features = in_dims[in_dims.len() - 1];

        // Flatten batch dimensions
        let x_flat = x.reshape(((), in_features))?;

        // Matrix multiply: x @ weight.T
        // VarBuilder for GGUF loads weights in [out_features, in_features] format.
        // So we transpose to [in_features, out_features] for matmul.
        let out = x_flat.matmul(&weight.t()?)?;

        // Reshape back
        let mut out_shape: Vec<usize> = batch_dims.to_vec();
        out_shape.push(out.dim(1)?);
        out.reshape(out_shape)
    }
}

/// RMS Layer Normalization
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new_from_quantized(weight: Arc<QTensor>, eps: f64, device: &Device) -> Result<Self> {
        let weight = weight.dequantize(device)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dim = x.dim(candle_core::D::Minus1)? as f64;
        let ms = x
            .sqr()?
            .sum_keepdim(candle_core::D::Minus1)?
            .affine(1.0 / dim, 0.0)?;
        // add eps as scalar via broadcasting
        let eps_tensor = Tensor::new(&[self.eps as f32], x.device())?.broadcast_as(ms.shape())?;
        let denom = ms.broadcast_add(&eps_tensor)?.sqrt()?;
        let mut ys = x.broadcast_div(&denom)?;

        // Broadcast weight over leading dims
        let rank = ys.rank();
        let mut shape = vec![1usize; rank];
        shape[rank - 1] = self.weight.dims1()?;
        let w = self.weight.reshape(shape)?;
        ys = ys.broadcast_mul(&w)?;

        Ok(ys)
    }
}

/// T5 Attention layer
struct T5Attention {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    relative_attention_bias: Option<Arc<QTensor>>,
    num_heads: usize,
    d_kv: usize,
}

impl T5Attention {
    fn new(vb: &mut Reader, block_idx: usize, config: &T5EncoderConfig) -> Result<Self> {
        let prefix = format!("enc.blk.{}", block_idx);

        let q = QLinear::new(vb.get(
            (config.num_heads * config.d_kv, config.d_model),
            &format!("{}.attn_q.weight", prefix),
        )?);
        let k = QLinear::new(vb.get(
            (config.num_heads * config.d_kv, config.d_model),
            &format!("{}.attn_k.weight", prefix),
        )?);
        let v = QLinear::new(vb.get(
            (config.num_heads * config.d_kv, config.d_model),
            &format!("{}.attn_v.weight", prefix),
        )?);
        let o = QLinear::new(vb.get(
            (config.d_model, config.num_heads * config.d_kv),
            &format!("{}.attn_o.weight", prefix),
        )?);

        // Relative position bias only in first layer - shape (32, 64) = (num_buckets, num_heads)
        let relative_attention_bias = if block_idx == 0 {
            Some(vb.get(
                (config.relative_attention_num_buckets, config.num_heads),
                &format!("{}.attn_rel_b.weight", prefix),
            )?)
        } else {
            None
        };

        Ok(Self {
            q,
            k,
            v,
            o,
            relative_attention_bias,
            num_heads: config.num_heads,
            d_kv: config.d_kv,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Compute Q, K, V
        let q = self.q.forward(hidden_states)?;
        let k = self.k.forward(hidden_states)?;
        let v = self.v.forward(hidden_states)?;

        // Reshape to [batch, heads, seq, d_kv]
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;

        // Attention scores
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?;

        // Add position bias
        let (mut scores, position_bias_out) = if let Some(bias) = position_bias {
            (scores.broadcast_add(bias)?, Some(bias.clone()))
        } else if let Some(ref rel_bias) = self.relative_attention_bias {
            let bias = self.compute_position_bias(seq_len, hidden_states.device(), rel_bias)?;

            let scores = scores.broadcast_add(&bias)?;
            (scores, Some(bias))
        } else {
            (scores, None)
        };

        if let Some(mask) = attention_mask {
            scores = scores.broadcast_add(mask)?;
        }

        // Softmax and apply to values
        // Fallback to CPU for softmax if CUDA implementation is missing or unstable
        let scores_cpu = scores.to_device(&Device::Cpu)?;
        let attn_weights_cpu = candle_nn::ops::softmax_last_dim(&scores_cpu)?;
        let attn_weights = attn_weights_cpu.to_device(hidden_states.device())?;

        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.d_kv,
        ))?;

        // Output projection
        let output = self.o.forward(&attn_output)?;

        Ok((output, position_bias_out))
    }

    fn compute_position_bias(
        &self,
        seq_len: usize,
        device: &Device,
        rel_bias: &Arc<QTensor>,
    ) -> Result<Tensor> {
        // Compute position indices on CPU to avoid CUDA kernel issues with I64 operations
        let cpu = Device::Cpu;

        // Create position indices on CPU
        let context_position = Tensor::arange(0u32, seq_len as u32, &cpu)?;
        let memory_position = Tensor::arange(0u32, seq_len as u32, &cpu)?;

        // Compute relative position: memory - context (matches transformers)
        let context_position = context_position.to_dtype(DType::I64)?.unsqueeze(1)?;
        let memory_position = memory_position.to_dtype(DType::I64)?.unsqueeze(0)?;
        let relative_position = memory_position.broadcast_sub(&context_position)?;

        // Compute buckets (on CPU)
        let buckets = self.relative_position_bucket(&relative_position, 32, 128)?;

        // Lookup bias values - weights are [num_buckets, num_heads]
        let bias_weights = rel_bias.dequantize(&cpu)?;

        let buckets_flat = buckets.flatten_all()?;
        // index_select on dim 0: [selected, num_heads]
        let values = bias_weights.index_select(&buckets_flat, 0)?;
        // Reshape to [seq_len, seq_len, num_heads]
        let values = values.reshape((seq_len, seq_len, self.num_heads))?;
        // Permute to [num_heads, seq_len, seq_len]
        let bias = values.permute((2, 0, 1))?;

        // Add batch dimension and move to target device
        let bias = bias.unsqueeze(0)?;
        bias.to_device(device)
    }

    fn relative_position_bucket(
        &self,
        relative_position: &Tensor,
        num_buckets: usize,
        max_distance: usize,
    ) -> Result<Tensor> {
        // Simplified bucket computation for encoder (bidirectional)
        let relative_position = relative_position.to_dtype(DType::I64)?;

        let num_buckets_i = num_buckets as i64;
        let half_buckets = num_buckets_i / 2;

        // Handle positive positions (forward-looking)
        let abs_pos = relative_position.abs()?;

        // Compute bucket for small positions
        let max_exact = half_buckets / 2;
        let is_small = abs_pos.lt(max_exact)?;

        // For larger positions, use log scale
        let max_distance_f = max_distance as f32;
        let max_exact_f = max_exact as f32;
        let abs_pos_f = abs_pos.to_dtype(DType::F32)?;

        // log_ratio = log(abs_pos / max_exact) / log(max_distance / max_exact)
        let max_exact_tensor =
            Tensor::new(&[max_exact_f], abs_pos_f.device())?.broadcast_as(abs_pos_f.shape())?;
        let ratio = abs_pos_f.broadcast_div(&max_exact_tensor)?;
        let log_ratio = ratio.log()?;
        let log_max = (max_distance_f / max_exact_f).ln();
        let log_max_tensor =
            Tensor::new(&[log_max], log_ratio.device())?.broadcast_as(log_ratio.shape())?;
        let normalized_log = log_ratio.broadcast_div(&log_max_tensor)?;

        // log_bucket = normalized_log * (half_buckets - max_exact) + max_exact
        let bucket_range = (half_buckets - max_exact) as f32;
        let range_tensor = Tensor::new(&[bucket_range], normalized_log.device())?
            .broadcast_as(normalized_log.shape())?;
        let log_bucket = normalized_log.broadcast_mul(&range_tensor)?;
        let log_bucket = log_bucket.broadcast_add(&max_exact_tensor)?;

        // Clamp to valid range
        let max_bucket = (half_buckets - 1) as f32;
        let large_bucket = log_bucket.clamp(0f32, max_bucket)?.to_dtype(DType::I64)?;

        // Combine small and large buckets
        let abs_pos_i = abs_pos.to_dtype(DType::I64)?;
        let bucket = is_small.where_cond(&abs_pos_i, &large_bucket)?;

        // Add offset for positive positions (forward-looking memory relative to context)
        let is_positive = relative_position.gt(0i64)?;
        let offset = Tensor::new(&[half_buckets], relative_position.device())?
            .broadcast_as(bucket.shape())?;
        let bucket = is_positive.where_cond(&bucket.broadcast_add(&offset)?, &bucket)?;

        bucket.to_dtype(DType::U32)
    }
}

/// T5 Feed-Forward layer (gated)
struct T5FeedForward {
    up: QLinear,
    gate: QLinear,
    down: QLinear,
}

impl T5FeedForward {
    fn new(vb: &mut Reader, block_idx: usize, config: &T5EncoderConfig) -> Result<Self> {
        let prefix = format!("enc.blk.{}", block_idx);

        // Candle VarBuilder sees GGUF shapes as [out, in] for linear layers
        // ffn_up/gate: [10240, 4096], ffn_down: [4096, 10240]
        let up = QLinear::new(vb.get(
            (config.d_ff, config.d_model),
            &format!("{}.ffn_up.weight", prefix),
        )?);
        let gate = QLinear::new(vb.get(
            (config.d_ff, config.d_model),
            &format!("{}.ffn_gate.weight", prefix),
        )?);
        let down = QLinear::new(vb.get(
            (config.d_model, config.d_ff),
            &format!("{}.ffn_down.weight", prefix),
        )?);

        Ok(Self { up, gate, down })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Gated GeLU: down(gelu_new(gate(x)) * up(x))
        // T5-v1.1-xxl uses gelu_new (tanh approximation)
        let gate_out = self.gate.forward(x)?;

        let gate_out = gelu_new(&gate_out)?;
        let up_out = self.up.forward(x)?;
        let hidden = (gate_out * up_out)?;
        self.down.forward(&hidden)
    }
}

/// T5 Encoder Block
struct T5EncoderBlock {
    attention: T5Attention,
    attn_norm: RmsNorm,
    ffn: T5FeedForward,
    ffn_norm: RmsNorm,
}

impl T5EncoderBlock {
    fn new(
        vb: &mut Reader,
        block_idx: usize,
        config: &T5EncoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("enc.blk.{}", block_idx);

        let attention = T5Attention::new(vb, block_idx, config)?;
        let attn_norm = RmsNorm::new_from_quantized(
            vb.get((config.d_model,), &format!("{}.attn_norm.weight", prefix))?,
            config.layer_norm_epsilon,
            device,
        )?;
        let ffn = T5FeedForward::new(vb, block_idx, config)?;
        let ffn_norm = RmsNorm::new_from_quantized(
            vb.get((config.d_model,), &format!("{}.ffn_norm.weight", prefix))?,
            config.layer_norm_epsilon,
            device,
        )?;

        Ok(Self {
            attention,
            attn_norm,
            ffn,
            ffn_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Self-attention with pre-norm
        let normed = self.attn_norm.forward(hidden_states)?;

        let (attn_output, position_bias_out) =
            self.attention
                .forward(&normed, position_bias, attention_mask)?;
        let hidden_states = (hidden_states + attn_output)?;

        // Feed-forward with pre-norm
        let normed = self.ffn_norm.forward(&hidden_states)?;
        let ffn_output = self.ffn.forward(&normed)?;
        let hidden_states = (hidden_states + ffn_output)?;

        Ok((hidden_states, position_bias_out))
    }
}

pub fn encode(path: &Path, ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let config = T5EncoderConfig::t5_xxl();
    let mut file = File::open(path)?;
    let index = gguf_file::Content::read(&mut file)?;
    let mut reader = Reader { file, index };
    let device = &Device::Cpu;
    let info = reader
        .index
        .tensor_infos
        .get("token_embd.weight")
        .ok_or_else(|| candle_core::Error::Msg("missing T5 token table".into()))?;
    if info.shape.dims() != [config.vocab_size, config.d_model] {
        candle_core::bail!("invalid T5 token table shape");
    }
    let mut hidden =
        prompt_embeddings(&mut reader.file, reader.index.tensor_data_offset, info, ids)?;
    let (b, s) = mask.dims2()?;
    let mask = mask.reshape((b, 1, 1, s))?.to_dtype(DType::F32)?;
    let mask = ((Tensor::ones_like(&mask)? - mask)? * -1e9f64)?;
    let mut bias = None;
    for i in 0..config.num_layers {
        let block = T5EncoderBlock::new(&mut reader, i, &config, device)?;
        (hidden, bias) = block.forward(&hidden, bias.as_ref(), Some(&mask))?;
    }
    RmsNorm::new_from_quantized(
        reader.get((config.d_model,), "enc.output_norm.weight")?,
        config.layer_norm_epsilon,
        device,
    )?
    .forward(&hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use std::io::Cursor;

    #[test]
    fn sparse_quantized_rows_match_full_vocabulary_lookup() -> Result<()> {
        let cpu = &Device::Cpu;
        let info = gguf_file::TensorInfo {
            ggml_dtype: GgmlDType::Q8_0,
            shape: (4, 32).into(),
            offset: 9,
        };
        let mut bytes = vec![0; 16];
        for row in 0..4 {
            bytes.extend(0x3c00u16.to_le_bytes()); // Q8_0 scale = 1
            bytes.extend((0..32).map(|column| (row * 32 + column) as u8));
        }
        let mut cursor = Cursor::new(bytes);
        let ids = Tensor::from_vec(vec![3u32, 1, 3, 0], (2, 2), cpu)?;
        let full = info.read(&mut cursor, 7, cpu)?.dequantize(cpu)?;
        let expected = candle_nn::Embedding::new(full, 32).forward(&ids)?;
        let selected = prompt_embeddings(&mut cursor, 7, &info, &ids)?;
        assert_eq!(
            expected.flatten_all()?.to_vec1::<f32>()?,
            selected.flatten_all()?.to_vec1::<f32>()?
        );
        assert!(prompt_embeddings(&mut cursor, 7, &info, &Tensor::new(&[4u32], cpu)?).is_err());
        Ok(())
    }
}
