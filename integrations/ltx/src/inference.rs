use crate::weights::{self, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_video::{
    models::ltx_video::{
        configs::presets,
        ltx_transformer::LtxVideoTransformer3DModel,
        quantized_t5_encoder::QuantizedT5EncoderModel,
        scheduler::FlowMatchEulerDiscreteScheduler,
        t2v_pipeline::{LtxPipeline, VideoTransformer3D},
        vae::AutoencoderKLLtxVideo,
    },
    utils::deterministic_rng::Pcg32,
};
use std::{path::Path, time::Instant};

pub(crate) const WEIGHT_LIMIT: usize = 9 * 1024 * 1024 * 1024;
pub const WIDTH: usize = 384;
pub const HEIGHT: usize = 256;
const FPS: u32 = 24;

pub fn validate_frames(frames: usize) -> Result<()> {
    if !(9..=121).contains(&frames) || !(frames - 1).is_multiple_of(8) {
        return Err("frames must be 8n+1 in 9..=121".into());
    }
    Ok(())
}

pub(crate) fn coordinates(
    frames: usize,
    height: usize,
    width: usize,
    fps: u32,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut values = Vec::with_capacity(frames * height * width * 3);
    for f in 0..frames {
        for y in 0..height {
            for x in 0..width {
                values.extend([
                    ((f * 8 + 1).saturating_sub(8)) as f32 / fps as f32,
                    (y * 32) as f32,
                    (x * 32) as f32,
                ]);
            }
        }
    }
    Tensor::from_vec(values, (1, frames * height * width, 3), device)
}

pub(crate) fn encode_prompt(prompt: &str, models: &Path) -> Result<(Tensor, Tensor)> {
    encode_prompt_tokens(prompt, models, 128)
}

pub(crate) fn encode_prompt_tokens(
    prompt: &str,
    models: &Path,
    tokens: usize,
) -> Result<(Tensor, Tensor)> {
    if !matches!(tokens, 128 | 256) {
        return Err("supported T5 lengths:128 or256".into());
    }
    let mut tokenizer =
        tokenizers::Tokenizer::from_file(models.join("text_encoder_gguf/tokenizer.json"))
            .map_err(|e| e.to_string())?;
    tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
    tokenizer.with_padding(None);
    let encoding = tokenizer.encode(prompt, true).map_err(|e| e.to_string())?;
    if encoding.len() > tokens {
        return Err("prompt exceeds configured token limit".into());
    }
    let mut ids = encoding.get_ids().to_vec();
    let mut mask = encoding.get_attention_mask().to_vec();
    ids.resize(tokens, 0);
    mask.resize(tokens, 0);
    let ids = Tensor::from_vec(ids, (1, tokens), &Device::Cpu)?;
    let mask = Tensor::from_vec(mask, (1, tokens), &Device::Cpu)?;
    let path = models.join("text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf");
    let embeddings = if weights::streaming()? {
        crate::stream_t5::encode(&path, &ids, &mask)?
    } else {
        QuantizedT5EncoderModel::load(path, &Device::Cpu)?.forward(&ids, Some(&mask))?
    };
    Ok((embeddings, mask.to_dtype(DType::F32)?))
} // T5 weights released before loading DiT.

pub(crate) fn denoise(
    model_path: &Path,
    embeddings: Tensor,
    mask: Tensor,
    device: &Device,
    rng: &mut Pcg32,
    shape: [usize; 3],
) -> Result<Tensor> {
    let [frames, height, width] = shape;
    let config = presets::v0_9_8_distilled_2b();
    let mut transformer: Box<dyn VideoTransformer3D> = if weights::streaming()? {
        Box::new(
            crate::reference::transformer::LtxVideoTransformer3DModel::from_disk(
                &config.transformer,
                model_path,
                device,
                true,
            )?,
        )
    } else {
        let tensors = weights::load(model_path, false, DType::BF16, device, WEIGHT_LIMIT)?;
        Box::new(LtxVideoTransformer3DModel::new(
            &config.transformer,
            VarBuilder::from_tensors(tensors, DType::BF16, device),
        )?)
    };
    let mut scheduler_config = config.scheduler;
    // Official LTX preserves explicit distilled sigmas without terminal stretching.
    scheduler_config.shift_terminal = None;
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_config)?;
    scheduler.set_timesteps(
        Some(7),
        device,
        config.inference.timesteps.as_deref(),
        Some(0.0),
        None,
    )?;
    let timesteps = scheduler.timesteps().to_vec1::<f32>()?;
    let latent_f = (frames - 1) / 8 + 1;
    let latent_h = height / 32;
    let latent_w = width / 32;
    let noise = rng.randn((1, 128, latent_f, latent_h, latent_w), device)?;
    let mut latents = LtxPipeline::pack_latents(&noise, 1, 1)?;
    let coords = coordinates(latent_f, latent_h, latent_w, FPS, device)?;
    let embeddings = embeddings.to_device(device)?.to_dtype(DType::BF16)?;
    let mask = mask.to_device(device)?.to_dtype(DType::BF16)?;
    for (index, timestep) in timesteps.into_iter().enumerate() {
        let start = Instant::now();
        let t = Tensor::full(timestep, (1,), device)?;
        let predicted = VideoTransformer3D::forward(
            transformer.as_mut(),
            &latents.to_dtype(DType::BF16)?,
            &embeddings,
            &t,
            &mask,
            latent_f,
            latent_h,
            latent_w,
            None,
            Some(&coords),
            None,
        )?;
        latents = scheduler
            .step(&predicted.to_dtype(DType::F32)?, timestep, &latents, None)?
            .prev_sample;
        device.synchronize()?;
        println!(
            "stage=denoise step={} elapsed_ms={}",
            index + 1,
            start.elapsed().as_millis()
        );
    }
    Ok(
        LtxPipeline::unpack_latents(&latents, latent_f, latent_h, latent_w, 1, 1)?
            .to_device(&Device::Cpu)?,
    )
} // DiT weights released before loading VAE; keep only tiny CPU latents.

pub(crate) fn decode(
    model_path: &Path,
    latents: Tensor,
    device: &Device,
    rng: &mut Pcg32,
) -> Result<Tensor> {
    decode_quality(model_path, latents, device, rng, false)
}

/// Compatibility wrapper retains the historical path; quality trials use
/// official normalized-latent noise before VAE denormalization.
pub(crate) fn decode_quality(
    model_path: &Path,
    latents: Tensor,
    device: &Device,
    rng: &mut Pcg32,
    normalized_noise: bool,
) -> Result<Tensor> {
    decode_quality_options(model_path, latents, device, rng, normalized_noise, true)
}

pub(crate) fn decode_quality_options(
    model_path: &Path,
    latents: Tensor,
    device: &Device,
    rng: &mut Pcg32,
    normalized_noise: bool,
    temporal_tiling: bool,
) -> Result<Tensor> {
    let config = presets::v0_9_8_distilled_2b();
    let small_tiles = !temporal_tiling && matches!(latents.dim(2)?, 7 | 10);
    let latents = latents.to_device(device)?;
    let noise = rng.randn(latents.shape(), device)?;
    let latents = if normalized_noise {
        mix_decode_noise(&latents, &noise)?
    } else {
        latents
    };
    let prepare = |mean: &Tensor, std: &Tensor| -> candle_core::Result<Tensor> {
        let latents = LtxPipeline::denormalize_latents(
            &latents,
            mean,
            std,
            config.vae.scaling_factor as f32,
        )?;
        let latents = if normalized_noise {
            latents
        } else {
            mix_decode_noise(&latents, &noise)?
        }
        .to_dtype(DType::BF16)?;
        Ok(latents)
    };
    let timestep = Tensor::full(0.05_f32, (1,), device)?.to_dtype(DType::BF16)?;
    let video = if weights::streaming()? {
        let vae = crate::vae_half::Half::decoder(
            &config.vae,
            weights::vae_half(model_path, false, device)?,
            small_tiles,
        )?;
        vae.decode(&prepare(&vae.mean, &vae.std)?, &timestep, temporal_tiling)?
    } else {
        let tensors = weights::load(model_path, true, DType::BF16, device, WEIGHT_LIMIT)?;
        let mut vae = AutoencoderKLLtxVideo::new(
            config.vae.clone(),
            VarBuilder::from_tensors(tensors, DType::BF16, device),
        )?;
        vae.use_tiling = true;
        vae.use_slicing = true;
        vae.use_framewise_decoding = temporal_tiling;
        if small_tiles {
            vae.enable_tiling(Some(256), Some(256), None, Some(192), Some(192), None);
        }
        vae.decode(
            &prepare(vae.latents_mean(), vae.latents_std())?,
            Some(&timestep),
            false,
            false,
        )?
        .1
    };
    Ok(video)
}

/// Pack the selected frame before transfer so a strided view cannot copy the
/// backing storage for the entire clip. Keep the original RGB rounding on CPU.
pub(crate) fn frame_rgb(video: &Tensor, index: usize) -> std::io::Result<Vec<u8>> {
    let values = (|| -> candle_core::Result<Vec<f32>> {
        video
            .i((0, .., index, .., ..))?
            .permute((1, 2, 0))?
            .contiguous()?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()
    })()
    .map_err(std::io::Error::other)?;
    rgb_bytes(values)
}

pub(crate) fn mix_decode_noise(latents: &Tensor, noise: &Tensor) -> candle_core::Result<Tensor> {
    latents.affine(0.975, 0.0)? + noise.affine(0.025, 0.0)?
}

pub(crate) fn rgb_bytes(values: Vec<f32>) -> std::io::Result<Vec<u8>> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(std::io::Error::other("non-finite VAE output"));
    }
    Ok(values
        .into_iter()
        .map(|v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect())
}

pub fn generate(
    prompt: &str,
    models: &Path,
    output: &Path,
    ffmpeg: &Path,
    frames: usize,
) -> Result<()> {
    validate_frames(frames)?;
    aster_core::planner::validate_prompt(prompt)?;
    if output.exists() || output.extension().and_then(|v| v.to_str()) != Some("mp4") {
        return Err("output must be a new .mp4 path".into());
    }
    if !ffmpeg.is_file() {
        return Err("FFmpeg executable is missing".into());
    }
    let _monitor = crate::telemetry::Monitor::start();
    let device = Device::new_cuda(0)?; // No silent CPU fallback.
    let start = Instant::now();
    let weights = models.join("ltxv-2b-0.9.8-distilled.safetensors");
    println!("stage=integrity_check");
    weights::verify(
        &weights,
        6_340_744_492,
        "76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f",
    )?;
    weights::verify(
        &models.join("text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
        3_386_856_640,
        "b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae",
    )?;
    let stage = Instant::now();
    println!("stage=text_encoding device=cpu");
    let (embeddings, mask) = encode_prompt(prompt, models)?;
    println!(
        "stage=text_encoding elapsed_ms={}",
        stage.elapsed().as_millis()
    );
    let mut rng = Pcg32::new(42, 1442695040888963407);
    let stage = Instant::now();
    let latents = denoise(
        &weights,
        embeddings,
        mask,
        &device,
        &mut rng,
        [frames, HEIGHT, WIDTH],
    )?;
    println!(
        "stage=denoise_total elapsed_ms={}",
        stage.elapsed().as_millis()
    );
    let stage = Instant::now();
    let video = decode(&weights, latents, &device, &mut rng)?;
    if video.dims() != [1, 3, frames, HEIGHT, WIDTH] {
        return Err(format!("unexpected decoded video shape {:?}", video.dims()).into());
    }
    println!("stage=decode elapsed_ms={}", stage.elapsed().as_millis());
    let spec = aster_core::video::VideoSpec {
        width: WIDTH,
        height: HEIGHT,
        frames,
        fps_num: FPS,
        fps_den: 1,
    };
    aster_core::mp4::encode(ffmpeg, output, spec, |writer| {
        for frame in 0..frames {
            let pixels = frame_rgb(&video, frame)?;
            writer.write_frame(frame, &pixels)?;
        }
        Ok(())
    })?;
    println!(
        "kind=neural_video model=ltx-0.9.8-2b-distilled mode=FULL seed=42 frames={frames} width={WIDTH} height={HEIGHT} fps={FPS} elapsed_ms={} process_peak_working_set_bytes={:?}",
        start.elapsed().as_millis(),
        aster_core::memory::peak_working_set_bytes()?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn rgb_export_preserves_frame_order_and_bf16_values() {
        use candle_core::{DType, Device, Tensor};
        let clip = Tensor::from_vec(
            vec![-1f32, 1., 0., 0.5, 1., -1.],
            (1, 3, 2, 1, 1),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        assert_eq!(super::frame_rgb(&clip, 0).unwrap(), [0, 128, 255]);
        assert_eq!(super::frame_rgb(&clip, 1).unwrap(), [255, 191, 0]);
        assert!(super::frame_rgb(&clip, 2).is_err());
    }
    #[test]
    fn frame_limits_match_temporal_compression() {
        for frames in [0, 1, 8, 24, 122, usize::MAX] {
            assert!(super::validate_frames(frames).is_err());
        }
        for frames in [9, 25, 121] {
            assert!(super::validate_frames(frames).is_ok());
        }
    }
    #[test]
    fn invalid_decoder_values_cannot_be_encoded_as_black_pixels() {
        assert!(super::rgb_bytes(vec![f32::NAN]).is_err());
        assert!(super::rgb_bytes(vec![f32::INFINITY]).is_err());
        assert_eq!(
            super::rgb_bytes(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap(),
            [0, 0, 128, 255, 255]
        );
    }
    #[test]
    fn distilled_schedule_preserves_published_sigmas_and_euler_step() {
        use super::*;
        let config = presets::v0_9_8_distilled_2b();
        let mut scheduler_config = config.scheduler;
        scheduler_config.shift_terminal = None;
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_config).unwrap();
        let sigmas = config.inference.timesteps.unwrap();
        scheduler
            .set_timesteps(Some(7), &Device::Cpu, Some(&sigmas), Some(0.0), None)
            .unwrap();
        let actual = scheduler.sigmas().to_vec1::<f32>().unwrap();
        for (a, b) in actual.iter().zip(&sigmas) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(actual.last(), Some(&0.0));
        let t = scheduler.timesteps().to_vec1::<f32>().unwrap()[0];
        let sample = Tensor::new(&[2_f32], &Device::Cpu).unwrap();
        let prediction = Tensor::new(&[3_f32], &Device::Cpu).unwrap();
        let result = scheduler
            .step(&prediction, t, &sample, None)
            .unwrap()
            .prev_sample
            .to_vec1::<f32>()
            .unwrap();
        assert!((result[0] - (2.0 + 3.0 * (0.9937 - 1.0))).abs() < 1e-6);
    }
    #[test]
    fn causal_coordinates_are_not_frame_indices() {
        let tensor = super::coordinates(3, 1, 1, 24, &candle_core::Device::Cpu).unwrap();
        assert_eq!(
            tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [0., 0., 0., 1. / 24., 0., 0., 9. / 24., 0., 0.]
        );
    }
}
