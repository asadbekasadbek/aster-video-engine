use crate::{
    inference,
    weights::{self, Result},
};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_video::{
    models::ltx_video::{
        configs::presets,
        scheduler::FlowMatchEulerDiscreteScheduler,
        t2v_pipeline::{LtxPipeline, VideoTransformer3D},
        vae::AutoencoderKLLtxVideo,
    },
    utils::deterministic_rng::Pcg32,
};
use std::path::Path;

pub fn encode_frame(path: &Path, frame: &Tensor, device: &Device) -> Result<Tensor> {
    encode_frame_sized(path, frame, device, [inference::HEIGHT, inference::WIDTH])
}

pub fn encode_frame_sized(
    path: &Path,
    frame: &Tensor,
    device: &Device,
    size: [usize; 2],
) -> Result<Tensor> {
    if !matches!(size, [256, 384] | [384, 576] | [384, 704] | [544, 960]) {
        return Err("unsupported reference dimensions".into());
    }
    if frame.dims().len() != 5
        || frame.dim(0)? != 1
        || frame.dim(1)? != 3
        || !matches!(frame.dim(2)?, 1 | 9)
        || frame.dim(3)? != size[0]
        || frame.dim(4)? != size[1]
    {
        return Err("invalid reference frame shape".into());
    }
    let tensors = weights::load(path, true, DType::BF16, device, inference::WEIGHT_LIMIT)?;
    let vae = AutoencoderKLLtxVideo::new(
        presets::v0_9_8_distilled_2b().vae,
        VarBuilder::from_tensors(tensors, DType::BF16, device),
    )?;
    let (_, posterior) = vae.encode(
        &frame.to_device(device)?.to_dtype(DType::BF16)?,
        false,
        false,
    )?;
    let normalized = LtxPipeline::normalize_latents(
        &posterior.mode()?.to_dtype(DType::F32)?,
        vae.latents_mean(),
        vae.latents_std(),
        vae.scaling_factor as f32,
    )?;
    Ok(normalized.to_device(&Device::Cpu)?)
}

/// Replace the prefix with exact reference tokens. All suffix tokens remain model output.
fn preserve_prefix(updated: &Tensor, reference: &Tensor) -> candle_core::Result<Tensor> {
    let (b, s, c) = updated.dims3()?;
    let (rb, rs, rc) = reference.dims3()?;
    if b != rb || c != rc || rs >= s || rs == 0 {
        candle_core::bail!("invalid reference prefix shape");
    }
    Tensor::cat(&[reference, &updated.narrow(1, rs, s - rs)?], 1)
}

pub struct Request<'a> {
    pub model: &'a Path,
    pub text: &'a (Tensor, Tensor),
    pub reference: Option<&'a Tensor>,
    pub frames: usize,
}

pub fn denoise(request: Request<'_>, device: &Device, rng: &mut Pcg32) -> Result<Tensor> {
    denoise_sized(request, device, rng, [inference::HEIGHT, inference::WIDTH])
}

pub fn denoise_sized(
    request: Request<'_>,
    device: &Device,
    rng: &mut Pcg32,
    size: [usize; 2],
) -> Result<Tensor> {
    inference::validate_frames(request.frames)?;
    if !matches!(size, [256, 384] | [384, 576] | [384, 704] | [544, 960]) {
        return Err("unsupported reference dimensions".into());
    }
    let config = presets::v0_9_8_distilled_2b();
    let tensors = weights::load(
        request.model,
        false,
        DType::BF16,
        device,
        inference::WEIGHT_LIMIT,
    )?;
    let mut transformer = super::transformer::LtxVideoTransformer3DModel::new(
        &config.transformer,
        VarBuilder::from_tensors(tensors, DType::BF16, device),
    )?;
    let mut schedule = config.scheduler;
    schedule.shift_terminal = None;
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(schedule)?;
    scheduler.set_timesteps(
        Some(7),
        device,
        config.inference.timesteps.as_deref(),
        Some(0.0),
        None,
    )?;
    let f = (request.frames - 1) / 8 + 1;
    let h = size[0] / 32;
    let w = size[1] / 32;
    let mut latents = LtxPipeline::pack_latents(&rng.randn((1, 128, f, h, w), device)?, 1, 1)?;
    let reference = request
        .reference
        .map(|r| -> Result<Tensor> {
            if r.dims().len() != 5
                || r.dim(0)? != 1
                || r.dim(1)? != 128
                || r.dim(2)? == 0
                || r.dim(2)? >= f
                || r.dim(3)? != h
                || r.dim(4)? != w
            {
                return Err("reference latent shape mismatch".into());
            }
            Ok(LtxPipeline::pack_latents(
                &r.to_device(device)?.to_dtype(DType::F32)?,
                1,
                1,
            )?)
        })
        .transpose()?;
    if let Some(r) = &reference {
        latents = preserve_prefix(&latents, r)?;
    }
    let frozen_tokens = reference
        .as_ref()
        .map(|r| r.dim(1))
        .transpose()?
        .unwrap_or(0);
    let coords = inference::coordinates(f, h, w, 24, device)?;
    let embeddings = request.text.0.to_device(device)?.to_dtype(DType::BF16)?;
    let mask = request.text.1.to_device(device)?.to_dtype(DType::BF16)?;
    let mut frozen_error = 0_f32;
    let mut forwards = 0;
    for t in scheduler.timesteps().to_vec1::<f32>()? {
        let time = if reference.is_some() {
            let mut times = vec![t; f * h * w];
            times[..frozen_tokens].fill(0.0);
            Tensor::from_vec(times, (1, f * h * w), device)?
        } else {
            Tensor::full(t, (1,), device)?
        };
        let prediction = VideoTransformer3D::forward(
            &mut transformer,
            &latents.to_dtype(DType::BF16)?,
            &embeddings,
            &time,
            &mask,
            f,
            h,
            w,
            None,
            Some(&coords),
            None,
        )?;
        latents = scheduler
            .step(&prediction.to_dtype(DType::F32)?, t, &latents, None)?
            .prev_sample;
        if let Some(r) = &reference {
            latents = preserve_prefix(&latents, r)?;
            frozen_error = frozen_error.max(
                (&latents.narrow(1, 0, frozen_tokens)? - r)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?,
            );
        }
        forwards += 1;
    }
    device.synchronize()?;
    println!(
        "transformer_forwards={forwards} processed_tokens={} frozen_tokens_per_step={} frozen_max_abs_error={frozen_error} skipped_transformer_forwards=0",
        forwards * f * h * w,
        frozen_tokens
    );
    Ok(LtxPipeline::unpack_latents(&latents, f, h, w, 1, 1)?.to_device(&Device::Cpu)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hard_prefix_is_exact_while_other_tokens_change() {
        let r = Tensor::new(&[[[7_f32, 8.]]], &Device::Cpu).unwrap();
        let mut x = Tensor::zeros((1, 3, 2), DType::F32, &Device::Cpu).unwrap();
        for _ in 0..7 {
            x = preserve_prefix(&x.affine(1., 1.).unwrap(), &r).unwrap();
        }
        assert_eq!(
            x.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [7., 8., 7., 7., 7., 7.]
        );
        assert!(preserve_prefix(&r, &r).is_err());
    }
}
