//! Spatial latent upsampler from the Lightricks 0.9.8 checkpoint.
//! Three temporal kernel planes share a batched spatial convolution implementation.
use crate::weights::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use std::path::Path;

fn conv3(x: &Tensor, vb: VarBuilder, out: usize) -> candle_core::Result<Tensor> {
    let (b, c, t, h, w) = x.dims5()?;
    let weights = vb.get((out, c, 3, 3, 3), "weight")?;
    let padded = x.pad_with_zeros(2, 1, 1)?;
    let mut sum = Tensor::zeros((b * t, out, h, w), DType::F32, x.device())?;
    for k in 0..3 {
        let frames = padded
            .narrow(2, k, t)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * t, c, h, w))?;
        let kernel = weights.i((.., .., k, .., ..))?.contiguous()?;
        sum = (sum + frames.conv2d(&kernel, 1, 1, 1, 1)?.to_dtype(DType::F32)?)?;
    }
    sum.broadcast_add(
        &vb.get(out, "bias")?
            .reshape((1, out, 1, 1))?
            .to_dtype(DType::F32)?,
    )?
    .to_dtype(x.dtype())?
    .reshape((b, t, out, h, w))?
    .transpose(1, 2)?
    .contiguous()
}

fn norm(x: &Tensor, v: VarBuilder) -> candle_core::Result<Tensor> {
    let (b, c, _, _, _) = x.dims5()?;
    let g = x.to_dtype(DType::F32)?.reshape((b, 32, ()))?;
    let delta = g.broadcast_sub(&g.mean_keepdim(2)?)?;
    let scaled = delta
        .broadcast_div(&(delta.sqr()?.mean_keepdim(2)? + 1e-5)?.sqrt()?)?
        .reshape(x.shape())?;
    let param = |key| {
        v.get(c, key)?
            .to_dtype(DType::F32)?
            .reshape((1, c, 1, 1, 1))
    };
    scaled
        .broadcast_mul(&param("weight")?)?
        .broadcast_add(&param("bias")?)?
        .to_dtype(x.dtype())
}

fn residuals(mut x: Tensor, v: VarBuilder) -> candle_core::Result<Tensor> {
    for i in 0..4 {
        let block = v.pp(i);
        let y = norm(&conv3(&x, block.pp("conv1"), 512)?, block.pp("norm1"))?.silu()?;
        x = (x + norm(&conv3(&y, block.pp("conv2"), 512)?, block.pp("norm2"))?)?.silu()?;
    }
    Ok(x)
}

fn statistics(x: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
    let (b, c, _, _, _) = x.dims5()?;
    let flat = x.reshape((b, c, ()))?;
    let m = flat.mean_keepdim(2)?;
    let sd = flat.var_keepdim(2)?.sqrt()?;
    Ok((m.reshape((b, c, 1, 1, 1))?, sd.reshape((b, c, 1, 1, 1))?))
}

pub fn upscale(low: &Tensor, path: &Path, model: &Path, device: &Device) -> Result<Tensor> {
    // Safety: the verified local checkpoint files must remain immutable during this call.
    let v = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::BF16, device)? };
    let stats = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, device)? }
        .pp("vae.per_channel_statistics");
    let mean = stats
        .get(128, "mean-of-means")?
        .reshape((1, 128, 1, 1, 1))?;
    let std = stats.get(128, "std-of-means")?.reshape((1, 128, 1, 1, 1))?;
    let low = low.to_device(device)?.to_dtype(DType::F32)?;
    let x = low
        .broadcast_mul(&std)?
        .broadcast_add(&mean)?
        .to_dtype(DType::BF16)?;
    let x = norm(&conv3(&x, v.pp("initial_conv"), 512)?, v.pp("initial_norm"))?.silu()?;
    let x = residuals(x, v.pp("res_blocks"))?;
    let (b, c, t, h, w) = x.dims5()?;
    let x = x.transpose(1, 2)?.contiguous()?.reshape((b * t, c, h, w))?;
    let conv = candle_nn::conv2d(
        c,
        4 * c,
        3,
        candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        v.pp("upsampler.0"),
    )?;
    let x = candle_nn::ops::pixel_shuffle(&conv.forward(&x)?, 2)?
        .reshape((b, t, c, h * 2, w * 2))?
        .transpose(1, 2)?
        .contiguous()?;
    let x = residuals(x, v.pp("post_upsample_res_blocks"))?;
    let high = conv3(&x, v.pp("final_conv"), 128)?
        .to_dtype(DType::F32)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?;
    let (lm, ls) = statistics(&low)?;
    let (hm, hs) = statistics(&high)?;
    Ok(high
        .broadcast_sub(&hm)?
        .broadcast_div(&hs.clamp(1e-8, f32::MAX)?)?
        .broadcast_mul(&ls)?
        .broadcast_add(&lm)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temporal_kernel_uses_zero_padding_and_all_three_planes() -> candle_core::Result<()> {
        let device = &Device::Cpu;
        let mut data = vec![0f32; 27];
        for (k, coef) in [1., 2., 3.].iter().enumerate() {
            data[k * 9 + 4] = *coef;
        }
        let vars = std::collections::HashMap::from([
            (
                "weight".into(),
                Tensor::from_vec(data, (1, 1, 3, 3, 3), device)?,
            ),
            ("bias".into(), Tensor::new(&[1f32], device)?),
        ]);
        let input = Tensor::new(&[2f32, 5.], device)?.reshape((1, 1, 2, 1, 1))?;
        let out = conv3(
            &input,
            VarBuilder::from_tensors(vars, DType::F32, device),
            1,
        )?;
        assert_eq!(out.flatten_all()?.to_vec1::<f32>()?, [20., 13.]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires local approved 0.9.8 upscaler and 13B checkpoint, and CUDA"]
    fn real_cuda_upsampler_preserves_time_and_channel_statistics() -> Result<()> {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/ltxv-13b-0.9.8-dev");
        let d = Device::new_cuda(0)?;
        let x = Tensor::arange(0u32, 128 * 3 * 4 * 5, &d)?
            .to_dtype(DType::F32)?
            .affine(0.002, 0.)?
            .sin()?
            .reshape((1, 128, 3, 4, 5))?;
        let y = upscale(
            &x,
            &dir.join("ltxv-spatial-upscaler-0.9.8.safetensors"),
            &dir.join("ltxv-13b-0.9.8-dev.safetensors"),
            &d,
        )?;
        assert_eq!(y.dims(), [1, 128, 3, 8, 10]);
        let (xm, xs) = statistics(&x)?;
        let (ym, ys) = statistics(&y)?;
        for error in [(xm - ym)?, (xs - ys)?] {
            assert!(error.abs()?.max_all()?.to_scalar::<f32>()? < 1e-4);
        }
        Ok(())
    }
}
