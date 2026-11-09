// Adapted from FerrisMind/candle-video ebe43c414d2519535357113cad22d15fba0ce900.
// Copyright FerrisMind. Apache-2.0; see reference/LICENSE.
// Direction-specific weight ownership for the verified LTX 0.9.8 checkpoint.
// Convolutions use upstream modules. Tiling order and blending follow the pinned backend.
use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use candle_video::models::ltx_video::vae::{
    AutoencoderKLLtxVideoConfig, DiagonalGaussianDistribution, DownsampleType, LtxVideoDecoder3d,
    LtxVideoEncoder3d,
};

pub struct Half<M> {
    net: M,
    pub mean: Tensor,
    pub std: Tensor,
    spatial_compression_ratio: usize,
    temporal_compression_ratio: usize,
    tile_sample_min_height: usize,
    tile_sample_min_width: usize,
    tile_sample_min_num_frames: usize,
    tile_sample_stride_height: usize,
    tile_sample_stride_width: usize,
    tile_sample_stride_num_frames: usize,
}
impl<M> Half<M> {
    fn new(
        net: M,
        c: &AutoencoderKLLtxVideoConfig,
        vb: VarBuilder,
        small_tiles: bool,
    ) -> Result<Self> {
        let size = if small_tiles { 256 } else { 512 };
        let stride = if small_tiles { 192 } else { 384 };
        Ok(Self {
            net,
            mean: vb.get(c.latent_channels, "latents_mean")?,
            std: vb.get(c.latent_channels, "latents_std")?,
            spatial_compression_ratio: c.spatial_compression_ratio,
            temporal_compression_ratio: c.temporal_compression_ratio,
            tile_sample_min_height: size,
            tile_sample_min_width: size,
            tile_sample_stride_height: stride,
            tile_sample_stride_width: stride,
            tile_sample_min_num_frames: 16,
            tile_sample_stride_num_frames: 8,
        })
    }
    fn blend_h(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        // python: for x in range(blend): b[..., x] = a[..., -blend+x]*(1-x/blend) + b[..., x]*(x/blend) [file:1]
        let blend = blend_extent.min(a.dims5()?.4).min(b.dims5()?.4);
        if blend == 0 {
            return Ok(b.clone());
        }

        // w: (blend,) from 0..blend-1 divided by blend
        let w = Tensor::arange(0u32, blend as u32, b.device())?
            .to_dtype(DType::F32)?
            .affine(1.0 / (blend as f64), 0.0)?;
        let w = w.reshape((1, 1, 1, 1, blend))?.to_dtype(b.dtype())?;
        let one_minus = w.neg()?.affine(1.0, 1.0)?;

        // b_head: первые blend столбцов, b_tail: остаток
        let b_head = b.i((.., .., .., .., 0..blend))?;
        let b_tail = b.i((.., .., .., .., blend..))?;

        // a_tail: последние blend столбцов
        let aw = a.dims5()?.4;
        let a_tail = a.i((.., .., .., .., (aw - blend)..aw))?;

        // mixed = a_tail*(1-w) + b_head*w
        let mixed = a_tail
            .broadcast_mul(&one_minus)?
            .add(&b_head.broadcast_mul(&w)?)?;
        Tensor::cat(&[&mixed, &b_tail], 4)
    }

    fn blend_v(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        // python: for y in range(blend): b[..., y, :] = a[..., -blend+y, :]*(1-y/blend) + b[..., y, :]*(y/blend) [file:1]
        let blend = blend_extent.min(a.dims5()?.3).min(b.dims5()?.3);
        if blend == 0 {
            return Ok(b.clone());
        }

        let w = Tensor::arange(0u32, blend as u32, b.device())?
            .to_dtype(DType::F32)?
            .affine(1.0 / (blend as f64), 0.0)?;
        let w = w.reshape((1, 1, 1, blend, 1))?.to_dtype(b.dtype())?;
        let one_minus = w.neg()?.affine(1.0, 1.0)?;

        let b_head = b.i((.., .., .., 0..blend, ..))?;
        let b_tail = b.i((.., .., .., blend.., ..))?;

        let ah = a.dims5()?.3;
        let a_tail = a.i((.., .., .., (ah - blend)..ah, ..))?;

        let mixed = a_tail
            .broadcast_mul(&one_minus)?
            .add(&b_head.broadcast_mul(&w)?)?;
        Tensor::cat(&[&mixed, &b_tail], 3)
    }

    fn blend_t(&self, a: &Tensor, b: &Tensor, blend_extent: usize) -> Result<Tensor> {
        // python: for x in range(blend): b[..., x, :, :] = a[..., -blend+x, :, :]*(1-x/blend) + b[..., x, :, :]*(x/blend) [file:1]
        let blend = blend_extent.min(a.dims5()?.2).min(b.dims5()?.2);
        if blend == 0 {
            return Ok(b.clone());
        }

        let w = Tensor::arange(0u32, blend as u32, b.device())?
            .to_dtype(DType::F32)?
            .affine(1.0 / (blend as f64), 0.0)?;
        let w = w.reshape((1, 1, blend, 1, 1))?.to_dtype(b.dtype())?;
        let one_minus = w.neg()?.affine(1.0, 1.0)?;

        let b_head = b.i((.., .., 0..blend, .., ..))?;
        let b_tail = b.i((.., .., blend.., .., ..))?;

        let at = a.dims5()?.2;
        let a_tail = a.i((.., .., (at - blend)..at, .., ..))?;

        let mixed = a_tail
            .broadcast_mul(&one_minus)?
            .add(&b_head.broadcast_mul(&w)?)?;
        Tensor::cat(&[&mixed, &b_tail], 2)
    }
}
impl Half<LtxVideoEncoder3d> {
    pub fn encoder(c: &AutoencoderKLLtxVideoConfig, vb: VarBuilder) -> Result<Self> {
        let down = c
            .downsample_types
            .iter()
            .map(|s| DownsampleType::parse(s))
            .collect::<Vec<_>>();
        let net = LtxVideoEncoder3d::new(
            c.in_channels,
            c.latent_channels,
            &c.block_out_channels,
            &c.spatiotemporal_scaling,
            &c.layers_per_block,
            &down,
            c.patch_size,
            c.patch_size_t,
            c.resnet_eps,
            c.is_causal,
            vb.pp("encoder"),
        )?;
        Self::new(net, c, vb, false)
    }
    pub fn posterior_mode(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, _, h, w) = x.dims5()?;
        let moments = if h > self.tile_sample_min_height || w > self.tile_sample_min_width {
            self.tiled_encode(x, false)?
        } else {
            self.net.forward(x, false)?
        };
        DiagonalGaussianDistribution::new(&moments)?.mode()
    }
    fn tiled_encode(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        // python tiled_encode: loops in sample-space, blends in latent-space [file:1]
        let (_b, _c, _t, height, width) = x.dims5()?;

        let latent_height = height / self.spatial_compression_ratio;
        let latent_width = width / self.spatial_compression_ratio;

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;

        let tile_latent_stride_h = self.tile_sample_stride_height / self.spatial_compression_ratio;
        let tile_latent_stride_w = self.tile_sample_stride_width / self.spatial_compression_ratio;

        let blend_h = tile_latent_min_h.saturating_sub(tile_latent_stride_h);
        let blend_w = tile_latent_min_w.saturating_sub(tile_latent_stride_w);

        // rows[i][j] = encoder(tile)
        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        for i in (0..height).step_by(self.tile_sample_stride_height) {
            let mut row: Vec<Tensor> = Vec::new();
            for j in (0..width).step_by(self.tile_sample_stride_width) {
                let h_end = (i + self.tile_sample_min_height).min(height);
                let w_end = (j + self.tile_sample_min_width).min(width);
                let tile = x.i((.., .., .., i..h_end, j..w_end))?;
                let enc = self.net.forward(&tile, train)?;

                row.push(enc);
            }
            rows.push(row);
        }

        let mut prev_row_blended: Vec<Tensor> = Vec::new();
        let mut result_rows: Vec<Tensor> = Vec::with_capacity(rows.len());
        for (ri, row) in rows.iter().enumerate() {
            let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
            let mut curr_row_blended: Vec<Tensor> = Vec::with_capacity(row.len());
            for (cj, tile) in row.iter().enumerate() {
                let mut tile = tile.clone();

                if ri > 0 {
                    let above = &prev_row_blended[cj];
                    tile = self.blend_v(above, &tile, blend_h)?;
                }
                if cj > 0 {
                    let left = &curr_row_blended[cj - 1];
                    tile = self.blend_h(left, &tile, blend_w)?;
                }

                // Store fully blended tile for future neighbors
                curr_row_blended.push(tile.clone());

                // Keep only the non-overlapping part for concatenation
                let h_slice = tile_latent_stride_h.min(tile.dim(3)?);
                let w_slice = tile_latent_stride_w.min(tile.dim(4)?);
                let sliced_tile = tile.i((.., .., .., 0..h_slice, 0..w_slice))?;
                result_row.push(sliced_tile);
            }
            result_rows.push(Tensor::cat(&result_row, 4)?);
            prev_row_blended = curr_row_blended;
        }

        let enc = Tensor::cat(&result_rows, 3)?;
        enc.i((.., .., .., 0..latent_height, 0..latent_width))
    }
}
impl Half<LtxVideoDecoder3d> {
    pub fn decoder(
        c: &AutoencoderKLLtxVideoConfig,
        vb: VarBuilder,
        small_tiles: bool,
    ) -> Result<Self> {
        let net = LtxVideoDecoder3d::new(
            c.latent_channels,
            c.out_channels,
            &c.decoder_block_out_channels,
            &c.decoder_spatiotemporal_scaling,
            &c.decoder_layers_per_block,
            c.patch_size,
            c.patch_size_t,
            c.resnet_eps,
            c.decoder_causal,
            &c.decoder_inject_noise,
            c.timestep_conditioning,
            &c.decoder_upsample_residual,
            &c.decoder_upsample_factor,
            vb.pp("decoder"),
        )?;
        Self::new(net, c, vb, small_tiles)
    }
    pub fn decode(&self, z: &Tensor, time: &Tensor, temporal: bool) -> Result<Tensor> {
        let (_, _, t, h, w) = z.dims5()?;
        if temporal && t > self.tile_sample_min_num_frames / self.temporal_compression_ratio {
            self.temporal_tiled_decode(z, Some(time), false)
        } else if h > self.tile_sample_min_height / self.spatial_compression_ratio
            || w > self.tile_sample_min_width / self.spatial_compression_ratio
        {
            self.tiled_decode(z, Some(time), false)
        } else {
            self.net.forward(z, Some(time), false)
        }
    }
    fn tiled_decode(&self, z: &Tensor, temb: Option<&Tensor>, train: bool) -> Result<Tensor> {
        // python tiled_decode: loops in latent-space, blends/crops in sample-space [file:1]
        let (_b, _c, _t, height, width) = z.dims5()?;

        let sample_height = height * self.spatial_compression_ratio;
        let sample_width = width * self.spatial_compression_ratio;

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;

        let tile_latent_stride_h = self.tile_sample_stride_height / self.spatial_compression_ratio;
        let tile_latent_stride_w = self.tile_sample_stride_width / self.spatial_compression_ratio;

        let blend_h = self
            .tile_sample_min_height
            .saturating_sub(self.tile_sample_stride_height);
        let blend_w = self
            .tile_sample_min_width
            .saturating_sub(self.tile_sample_stride_width);

        // rows[i][j] = decoder(tile)
        let mut rows: Vec<Vec<Tensor>> = Vec::new();
        for i in (0..height).step_by(tile_latent_stride_h) {
            let mut row: Vec<Tensor> = Vec::new();
            for j in (0..width).step_by(tile_latent_stride_w) {
                let h_end = (i + tile_latent_min_h).min(height);
                let w_end = (j + tile_latent_min_w).min(width);
                let tile = z.i((.., .., .., i..h_end, j..w_end))?;
                let dec = self.net.forward(&tile, temb, train)?;
                row.push(dec);
            }
            rows.push(row);
        }

        let mut prev_row_blended: Vec<Tensor> = Vec::new();
        let mut result_rows: Vec<Tensor> = Vec::with_capacity(rows.len());
        for (ri, row) in rows.iter().enumerate() {
            let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
            let mut curr_row_blended: Vec<Tensor> = Vec::with_capacity(row.len());
            for (cj, tile) in row.iter().enumerate() {
                let mut tile = tile.clone();

                if ri > 0 {
                    let above = &prev_row_blended[cj];
                    tile = self.blend_v(above, &tile, blend_h)?;
                }
                if cj > 0 {
                    let left = &curr_row_blended[cj - 1];
                    tile = self.blend_h(left, &tile, blend_w)?;
                }

                // Store fully blended tile for future neighbors
                curr_row_blended.push(tile.clone());

                let h_slice = self.tile_sample_stride_height.min(tile.dim(3)?);
                let w_slice = self.tile_sample_stride_width.min(tile.dim(4)?);
                let sliced_tile = tile.i((.., .., .., 0..h_slice, 0..w_slice))?;
                result_row.push(sliced_tile);
            }
            result_rows.push(Tensor::cat(&result_row, 4)?);
            prev_row_blended = curr_row_blended;
        }

        let dec = Tensor::cat(&result_rows, 3)?;
        dec.i((.., .., .., 0..sample_height, 0..sample_width))
    }

    // ===== temporal tiling =====

    fn temporal_tiled_decode(
        &self,
        z: &Tensor,
        temb: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        // python temporal_tiled_decode: stride in latent time, blends in sample time [file:1]
        let (_b, _c, num_frames, _h, _w) = z.dims5()?;

        let num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1; // python formula [file:1]

        let tile_latent_min_h = self.tile_sample_min_height / self.spatial_compression_ratio;
        let tile_latent_min_w = self.tile_sample_min_width / self.spatial_compression_ratio;

        let tile_latent_min_t = self.tile_sample_min_num_frames / self.temporal_compression_ratio;
        let tile_latent_stride_t =
            self.tile_sample_stride_num_frames / self.temporal_compression_ratio;

        // Python: blend_num_frames = tile_sample_min - tile_sample_stride = 16 - 8 = 8
        let blend_t_sample = self
            .tile_sample_min_num_frames
            .saturating_sub(self.tile_sample_stride_num_frames);

        let mut row: Vec<Tensor> = Vec::new();
        for (loop_idx, i) in (0..num_frames).step_by(tile_latent_stride_t).enumerate() {
            let t_end = (i + tile_latent_min_t + 1).min(num_frames);
            let tile = z.i((.., .., i..t_end, .., ..))?;

            let decoded =
                if tile.dims5()?.3 > tile_latent_min_h || tile.dims5()?.4 > tile_latent_min_w {
                    self.tiled_decode(&tile, temb, train)?
                } else {
                    self.net.forward(&tile, temb, train)?
                };

            // Python: if i > 0: decoded = decoded[:, :, :-1, :, :]
            // Remove last sample frame from all tiles except first
            let decoded = if loop_idx > 0 {
                let t = decoded.dim(2)?;
                if t > 1 {
                    decoded.i((.., .., 0..(t - 1), .., ..))?
                } else {
                    decoded
                }
            } else {
                decoded
            };

            row.push(decoded);
        }

        // Python logic:
        // for i, tile in enumerate(row):
        //     if i > 0:
        //         tile = self.blend_t(row[i - 1], tile, blend_num_frames)
        //         tile = tile[:, :, :stride, :, :]  # Take FIRST stride frames
        //     else:
        //         tile = tile[:, :, :stride+1, :, :]  # First tile: stride+1 frames
        let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
        for (idx, tile) in row.iter().enumerate() {
            let tile = if idx > 0 {
                let blended = self.blend_t(&row[idx - 1], tile, blend_t_sample)?;
                // Take FIRST stride frames (not last!)
                let end = self.tile_sample_stride_num_frames.min(blended.dim(2)?);
                blended.i((.., .., 0..end, .., ..))?
            } else {
                // First tile: keep stride + 1 frames
                let end = (self.tile_sample_stride_num_frames + 1).min(tile.dim(2)?);
                tile.i((.., .., 0..end, .., ..))?
            };
            result_row.push(tile);
        }

        let dec = Tensor::cat(&result_row, 2)?;
        dec.i((.., .., 0..num_sample_frames, .., ..))
    }
}
