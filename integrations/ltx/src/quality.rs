//! Bounded one-second quality experiments; no scene/identity or efficiency claims.
use crate::{
    inference,
    weights::{self, Result},
};
use candle_core::Device;
use candle_video::utils::deterministic_rng::Pcg32;
use std::{fs::OpenOptions, io::Write, path::Path, process::Command, time::Instant};

#[derive(Debug, Clone, Copy)]
pub struct Profile {
    pub width: usize,
    pub height: usize,
    pub normalized_noise: bool,
    pub text_tokens: usize,
}
impl Profile {
    pub fn named(name: &str) -> Result<Self> {
        let base = Self {
            width: 384,
            height: 256,
            normalized_noise: true,
            text_tokens: 128,
        };
        Ok(match name {
            "legacy" => Self {
                normalized_noise: false,
                ..base
            },
            "corrected" => base,
            "native576" => Self {
                width: 576,
                height: 384,
                ..base
            },
            "native576-t5-256" | "text576-fullvae" => Self {
                width: 576,
                height: 384,
                text_tokens: 256,
                ..base
            },
            "text960-fullvae" => Self {
                width: 960,
                height: 640,
                text_tokens: 256,
                ..base
            },
            "reference960-fullvae" | "reference960-3s-fullvae" => Self {
                width: 960,
                height: 544,
                text_tokens: 256,
                ..base
            },
            "reference13b-704-fullvae"
            | "reference704"
            | "reference704-fullvae"
            | "reference704-3s-fullvae" => Self {
                width: 704,
                height: 384,
                text_tokens: 256,
                ..base
            },
            _ => {
                return Err(
                    "quality profiles:legacy|corrected|native576|native576-t5-256|reference13b-multiscale-2s|reference13b-704-fullvae|reference704|reference704-fullvae|reference704-3s-fullvae|reference960-fullvae|reference960-3s-fullvae|text576-fullvae|text960-fullvae"
                        .into(),
                );
            }
        })
    }
}

pub fn generate(
    prompt: &str,
    models: &Path,
    output: &Path,
    ffmpeg: &Path,
    profile: &str,
    seed: u64,
    image: Option<&Path>,
) -> Result<()> {
    if profile == "reference13b-multiscale-2s" {
        return crate::reference::multiscale::generate(
            prompt,
            models,
            output,
            ffmpeg,
            seed,
            image.ok_or("reference image required")?,
        );
    }
    let p = Profile::named(profile)?;
    let dev = profile == "reference13b-704-fullvae";
    if dev && !weights::streaming()? {
        return Err("13B requires ASTER_WEIGHT_MODE=stream".into());
    }
    let (name, hash, bytes) = if dev {
        (
            "ltxv-13b-0.9.8-dev",
            "56b39a874ef9d0c7ad4099954f0ee4fa7e600ce83a24b3736f9e12223e7c71be",
            28579183340u64,
        )
    } else {
        (
            "ltxv-2b-0.9.8-distilled",
            "76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f",
            6340744492,
        )
    };
    let sigmas = if dev {
        crate::reference::latent::dev_sigmas()
            .iter()
            .map(f32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    } else {
        "1,0.9937,0.9875,0.9812,0.9750,0.9094,0.7250".into()
    };
    let steps = if dev { 30 } else { 7 };
    let three_seconds = matches!(
        profile,
        "reference704-3s-fullvae" | "reference960-3s-fullvae"
    );
    let (model_frames, output_frames, tile) = if three_seconds {
        (73, 72, 256)
    } else {
        (25, 25, 512)
    };
    aster_core::planner::validate_prompt(prompt)?;
    if profile.starts_with("text") && image.is_some() {
        return Err("text-only profile cannot use an image".into());
    }
    if profile.starts_with("reference") && image.is_none() {
        return Err("reference profiles require --image".into());
    }
    if output.exists()
        || output.extension().and_then(|s| s.to_str()) != Some("mp4")
        || !ffmpeg.is_file()
    {
        return Err("need new MP4 and installed FFmpeg".into());
    }
    // Conservative admission, not a claim to account for every CUDA allocation.
    let mut query = Command::new("nvidia-smi");
    query.args([
        "--id=0",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        query.creation_flags(0x08000000);
    }
    let free = query.output()?;
    if !free.status.success() {
        return Err("cannot verify available GPU memory".into());
    }
    let free: u64 = std::str::from_utf8(&free.stdout)?.trim().parse()?;
    let required = if p.width > 384 { 10240 } else { 7168 };
    if free < required {
        return Err(format!(
            "quality profile requires {required}MiB free headroom; observed {free}"
        )
        .into());
    }
    let mut record = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.with_extension("params.txt"))?;
    writeln!(
        record,
        "status=planned\nprofile={profile}\nprompt={prompt:?}\nseed={seed}\nrng=PCG32-stream1442695040888963407-not-torch-generator\nmodel={name}\nmodel_sha256={hash}\ntext_encoder=t5-v1_1-xxl-encoder-Q5_K_M.gguf\ntext_encoder_sha256=b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae\ntext_max_tokens={}\nwidth={}\nheight={}\nframes={output_frames}\nmodel_frames={model_frames}\nfps=24\nprecision_dit_vae=BF16\nprecision_scheduler=F32\nweight_quantization=none-for-DiT-VAE,T5-Q5_K_M\nsigmas={sigmas},terminal0\nscheduler=FlowMatchEuler-explicit-sigmas-no-terminal-stretch\nsteps={steps}\nCFG=1\nSTG=0\nstochastic_sampling=false\nnegative_prompt=unused-CFG1\ndecode_timestep=0.05\ndecode_noise_scale=0.025\nnoise_before_denormalization={}\nvae=checkpoint-bundled-per-channel-normalization\nvae_spatial_tile={tile}\nvae_temporal_tile=16-stride8\nvae_slicing=true\npasses=1\nlearned_upscaler=false\nprompt_enhancement=false\ninitial_free_gpu_mib={free}\ntelemetry=sampled-device-wide-stdout-log\n",
        p.text_tokens, p.width, p.height, p.normalized_noise
    )?;
    let _monitor = crate::telemetry::Monitor::start();
    writeln!(
        record,
        "weight_mode={}",
        if weights::streaming()? {
            "stream"
        } else {
            "resident"
        }
    )?;
    let device = Device::new_cuda(0)?;
    let start = Instant::now();
    let model = if dev {
        std::env::var_os("ASTER_13B_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                models
                    .with_file_name("ltxv-13b-0.9.8-dev")
                    .join(format!("{name}.safetensors"))
            })
    } else {
        models.join(format!("{name}.safetensors"))
    };
    weights::verify(&model, bytes, hash)?;
    weights::verify(
        &models.join("text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
        3_386_856_640,
        "b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae",
    )?;
    let generation = Instant::now();
    let (embeddings, mask) = inference::encode_prompt_tokens(prompt, models, p.text_tokens)?;
    println!(
        "stage=text_encoded process_peak_working_set_bytes={:?}",
        aster_core::memory::peak_working_set_bytes()?
    );
    let mut rng = Pcg32::new(seed, 1442695040888963407);
    let latents = if let Some(image) = image {
        let input = load_image(ffmpeg, image, p.height, p.width)?;
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(image)?;
        writeln!(
            record,
            "mode=image-to-video\nimage_path={}\nimage_sha256={:x}\nimage_resize=lanczos-aspect-fill-center-crop\nconditioning_strength=1\nimage_conditioning_noise=0",
            image.display(),
            Sha256::digest(bytes)
        )?;
        let reference = crate::reference::latent::encode_frame_sized(
            &model,
            &input,
            &device,
            [p.height, p.width],
        )?;
        crate::reference::latent::denoise_model(
            crate::reference::latent::Request {
                model: &model,
                text: &(embeddings, mask),
                reference: Some(&reference),
                frames: model_frames,
            },
            &device,
            &mut rng,
            [p.height, p.width],
            dev,
        )?
    } else {
        writeln!(record, "mode=text-to-video")?;
        inference::denoise(
            &model,
            embeddings,
            mask,
            &device,
            &mut rng,
            [model_frames, p.height, p.width],
        )?
    };
    let temporal_tiling = !profile.ends_with("-fullvae");
    writeln!(record, "vae_temporal_tiling_enabled={temporal_tiling}")?;
    let video = inference::decode_quality_options(
        &model,
        latents,
        &device,
        &mut rng,
        p.normalized_noise,
        temporal_tiling,
    )?;
    if video.dims() != [1, 3, model_frames, p.height, p.width] {
        return Err("quality decoded shape mismatch".into());
    }
    let spec = aster_core::video::VideoSpec {
        width: p.width,
        height: p.height,
        frames: output_frames,
        fps_num: 24,
        fps_den: 1,
    };
    aster_core::mp4::encode(ffmpeg, output, spec, |writer| {
        for i in 0..output_frames {
            writer.write_frame(i, &inference::frame_rgb(&video, i)?)?;
        }
        Ok(())
    })?;
    let summary = format!(
        "status=generated_not_quality_accepted elapsed_ms={} generation_ms={} process_peak_working_set_bytes={:?}",
        start.elapsed().as_millis(),
        generation.elapsed().as_millis(),
        aster_core::memory::peak_working_set_bytes()?
    );
    println!(
        "{summary} profile={profile} seed={seed} width={} height={} frames={output_frames}",
        p.width, p.height
    );
    writeln!(record, "{summary}")?;
    record.sync_all()?;
    Ok(())
}

pub(crate) fn load_image(
    ffmpeg: &Path,
    image: &Path,
    height: usize,
    width: usize,
) -> Result<candle_core::Tensor> {
    let metadata = std::fs::metadata(image)?;
    if !metadata.is_file() || metadata.len() > 20 * 1024 * 1024 {
        return Err("reference must be an image file <=20MiB".into());
    }
    let output = Command::new(ffmpeg).args(["-v", "error", "-nostdin", "-i"]).arg(image)
        .args(["-vf", &format!("scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height}"), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]).output()?;
    if !output.status.success() || output.stdout.len() != width * height * 3 {
        return Err("reference image decode failed or wrong extent".into());
    }
    image_tensor(&output.stdout, height, width)
}

fn image_tensor(bytes: &[u8], height: usize, width: usize) -> Result<candle_core::Tensor> {
    if !matches!(
        [height, width],
        [224, 448] | [448, 896] | [256, 384] | [384, 576] | [384, 704] | [544, 960]
    ) || bytes.len() != height * width * 3
    {
        return Err("invalid image dimensions/extent".into());
    }
    let pixels: Vec<f32> = bytes.iter().map(|v| *v as f32 / 127.5 - 1.).collect();
    Ok(
        candle_core::Tensor::from_vec(pixels, (height, width, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .unsqueeze(2)?
            .contiguous()?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    #[test]
    fn image_input_shape_range_and_extent() {
        let mut pixels = vec![0_u8; 256 * 384 * 3];
        pixels[0] = 255;
        let t = image_tensor(&pixels, 256, 384).unwrap();
        assert_eq!(t.dims(), [1, 3, 1, 256, 384]);
        assert_eq!(t.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0], 1.);
        assert_eq!(t.min_all().unwrap().to_scalar::<f32>().unwrap(), -1.);
        assert!(image_tensor(&pixels[..12], 256, 384).is_err());
        assert!(image_tensor(&pixels, 384, 256).is_err());
    }
    #[test]
    fn profiles_are_bounded_and_noise_is_mixed_in_normalized_space() {
        assert!(Profile::named("13b").is_err());
        assert_eq!(Profile::named("native576").unwrap().width, 576);
        let x = Tensor::new(&[2_f32], &Device::Cpu).unwrap();
        let n = Tensor::new(&[4_f32], &Device::Cpu).unwrap();
        let official = inference::mix_decode_noise(&x, &n)
            .unwrap()
            .affine(3., 5.)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        let old = inference::mix_decode_noise(&x.affine(3., 5.).unwrap(), &n)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((official - 11.15).abs() < 1e-5);
        assert!((old - 10.825).abs() < 1e-5);
    }
}
