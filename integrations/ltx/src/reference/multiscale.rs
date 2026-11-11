//! 13B Dev two-pass image-conditioned inference; stages mirror the upstream 0.9.8 YAML.
use super::{latent, transformer::LtxVideoTransformer3DModel, upscaler};
use crate::{
    inference, quality,
    weights::{self, Result},
};
use candle_core::{DType, Device, Tensor};
use candle_video::{
    models::ltx_video::{configs::presets, t2v_pipeline::LtxPipeline},
    utils::deterministic_rng::Pcg32,
};
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
const NEGATIVE: &str = "worst quality, inconsistent motion, blurry, jittery, distorted";
type Text = (Tensor, Tensor);

fn guidance(sigma: f32, high: bool) -> (f64, f64, f64, &'static [usize]) {
    if high {
        return (1., 1., 1., &[27]);
    }
    let i = [1., 0.996, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180]
        .iter()
        .position(|&s| s <= sigma)
        .unwrap_or(6);
    let blocks: &[&[usize]] = &[
        &[],
        &[11, 25, 35, 39],
        &[22, 35, 39],
        &[28],
        &[28],
        &[28],
        &[28],
    ];
    (
        [1., 1., 6., 8., 6., 1., 1.][i],
        [0., 0., 4., 4., 4., 2., 1.][i],
        [1., 1., 0.5, 0.5, 1., 1., 1.][i],
        blocks[i],
    )
}

struct Pass<'a> {
    model: &'a Path,
    device: &'a Device,
    positive: &'a Text,
    negative: &'a Text,
    photo: &'a Path,
    ffmpeg: &'a Path,
}
impl Pass<'_> {
    fn run(&self, initial: Option<Tensor>, rng: &mut Pcg32) -> Result<Tensor> {
        let high = initial.is_some();
        let (h, w) = if high { (14, 28) } else { (7, 14) };
        let d = self.device;
        let depth = 7;
        let size = [h * 32, w * 32];
        let image = quality::load_image(self.ffmpeg, self.photo, size[0], size[1])?;
        let reference = latent::encode_frame_sized(self.model, &image, d, size)?;
        let reference = LtxPipeline::pack_latents(&reference.to_device(d)?, 1, 1)?;
        let config = presets::v0_9_8_dev_13b();
        let mut net =
            LtxVideoTransformer3DModel::from_disk(&config.transformer, self.model, d, false)?;
        let steps = latent::dev_sigmas();
        let schedule = if high { &steps[17..] } else { &steps[..27] };
        let s = f64::from(schedule[0]);
        let noise = rng.randn((1, 128, depth, h, w), d)?;
        let start = match initial {
            Some(x) => (x.affine(1. - s, 0.)? + noise.affine(s, 0.)?)?,
            None => noise,
        };
        let mut x = LtxPipeline::pack_latents(&start, 1, 1)?;
        let coords = inference::coordinates(depth, h, w, 24, d)?;
        let ref_len = h * w;
        let tail = |x: &Tensor| x.narrow(1, ref_len, (depth - 1) * ref_len);
        for (i, &sigma) in schedule.iter().enumerate() {
            let s = f64::from(sigma);
            let ref_noise = rng.randn(reference.shape(), d)?.affine(0.15 * s * s, 0.)?;
            let noisy_ref = (&reference + ref_noise)?;
            x = Tensor::cat(&[noisy_ref, tail(&x)?], 1)?;
            let input = x.to_dtype(DType::BF16)?;
            let mut time = vec![sigma * 1000.; depth * ref_len];
            time[..ref_len].fill(0.);
            let time = Tensor::from_vec(time, (1, depth * ref_len), d)?;
            let mut forward = |text: &Text, blocks: &[usize]| -> candle_core::Result<Tensor> {
                net.value_blocks.clear();
                net.value_blocks.extend_from_slice(blocks);
                net.forward(
                    &input,
                    &text.0,
                    &time,
                    Some(&text.1),
                    depth,
                    h,
                    w,
                    None,
                    Some(&coords),
                    None,
                )?
                .to_dtype(DType::F32)
            };
            let (cfg, stg, rescale, blocks) = guidance(sigma, high);
            let positive = forward(self.positive, &[])?;
            let mut pred = positive.clone();
            if cfg > 1. {
                let neg = forward(self.negative, &[])?;
                let alpha = ((&positive * &neg)?.sum_all()?.to_scalar::<f32>()?)
                    / (neg.sqr()?.sum_all()?.to_scalar::<f32>()? + 1e-8);
                let base = neg.affine(f64::from(alpha), 0.)?;
                pred = (&base + (&positive - &base)?.affine(cfg, 0.)?)?;
            }
            if stg > 0. {
                let perturbed = forward(self.positive, blocks)?;
                pred = (pred + (&positive - perturbed)?.affine(stg, 0.)?)?;
                if rescale != 1. {
                    let std = |a: &Tensor| a.flatten_all()?.var(0)?.sqrt()?.to_scalar::<f32>();
                    pred = pred.affine(
                        1. - rescale + rescale * f64::from(std(&positive)? / std(&pred)?.max(1e-8)),
                        0.,
                    )?;
                }
            }
            let next = f64::from(schedule.get(i + 1).copied().unwrap_or(0.));
            let updated = (tail(&x)? + tail(&pred)?.affine(next - s, 0.)?)?;
            x = Tensor::cat(&[x.narrow(1, 0, ref_len)?, updated], 1)?;
            println!(
                "scale={} step={}/{} sigma={sigma}",
                if high { "high" } else { "low" },
                i + 1,
                schedule.len()
            );
        }
        Ok(LtxPipeline::unpack_latents(&x, depth, h, w, 1, 1)?)
    }
}

pub fn generate(
    prompt: &str,
    models: &Path,
    output: &Path,
    ffmpeg: &Path,
    seed: u64,
    photo: &Path,
) -> Result<()> {
    if !weights::streaming()?
        || output.exists()
        || output.extension().and_then(|s| s.to_str()) != Some("mp4")
    {
        return Err("multiscale requires streamed weights and a new MP4 path".into());
    }
    aster_core::planner::validate_prompt(prompt)?;
    let d = Device::new_cuda(0)?;
    let free = crate::prefetch::free_vram(&d)?;
    if free < 8 * 1024usize.pow(3) {
        return Err("multiscale requires 8 GiB free GPU memory".into());
    }
    let model = std::env::var_os("ASTER_13B_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            models
                .with_file_name("ltxv-13b-0.9.8-dev")
                .join("ltxv-13b-0.9.8-dev.safetensors")
        });
    let up = model.with_file_name("ltxv-spatial-upscaler-0.9.8.safetensors");
    weights::verify(
        &up,
        505024432,
        "5b076031c6f860db9037a54f3bb819f10bfb5532ea26a6d30062292428a0c208",
    )?;
    weights::verify(
        &model,
        28579183340,
        "56b39a874ef9d0c7ad4099954f0ee4fa7e600ce83a24b3736f9e12223e7c71be",
    )?;
    weights::verify(
        &models.join("text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
        3386856640,
        "b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae",
    )?;
    let mut record = std::fs::File::create_new(output.with_extension("params.txt"))?;
    writeln!(
        record,
        "profile=reference13b-multiscale-2s\nmodel={}\nupscaler={}\nphoto={}\nprompt={prompt:?}\nnegative_prompt={NEGATIVE}\nseed={seed}\nrng=PCG32\ntext=T5-XXL-Q5_K_M-256\nprompt_enhancement=false\nprecision=BF16\nscheduler=LinearQuadratic30\nlow=448x224 steps=0..26\nhigh=896x448 steps=17..29\nCFG-star=true STG=attention_values rescale=dynamic AdaIN=true\nimage_cond_noise=0.15 decode_timestep=0.05 decode_noise=0.025\noutput=704x384 fps=24 model_frames=49 encoded_frames=48",
        model.display(),
        up.display(),
        photo.display()
    )?;
    let _monitor = crate::telemetry::Monitor::start();
    let start = Instant::now();
    let encode = |s: &str| -> Result<Text> {
        let (x, m) = inference::encode_prompt_tokens(s, models, 256)?;
        Ok((
            x.to_device(&d)?.to_dtype(DType::BF16)?,
            m.to_device(&d)?.to_dtype(DType::BF16)?,
        ))
    };
    let positive = encode(prompt)?;
    let negative = encode(NEGATIVE)?;
    let pass = Pass {
        model: &model,
        device: &d,
        photo,
        ffmpeg,
        positive: &positive,
        negative: &negative,
    };
    let mut rng = Pcg32::new(seed, 1442695040888963407);
    let low = pass.run(None, &mut rng)?;
    let high = upscaler::upscale(&low, &up, &model, &d)?;
    let final_latent = pass.run(Some(high), &mut rng)?;
    let video = inference::decode_quality_options(&model, final_latent, &d, &mut rng, true, false)?;
    if video.dims() != [1, 3, 49, 448, 896] {
        return Err("unexpected decoded shape".into());
    }
    let resized = video
        .squeeze(0)?
        .transpose(0, 1)?
        .contiguous()?
        .to_dtype(DType::F32)?
        .upsample_bilinear2d(384, 704, false)?
        .transpose(0, 1)?
        .unsqueeze(0)?;
    let spec = aster_core::video::VideoSpec {
        width: 704,
        height: 384,
        frames: 48,
        fps_num: 24,
        fps_den: 1,
    };
    aster_core::mp4::encode(ffmpeg, output, spec, |writer| {
        for i in 0..48 {
            writer.write_frame(i, &inference::frame_rgb(&resized, i)?)?;
        }
        Ok(())
    })?;
    writeln!(
        record,
        "generation_ms={} peak_process_bytes={:?}",
        start.elapsed().as_millis(),
        aster_core::memory::peak_working_set_bytes()?
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lookup_uses_next_lower_threshold_and_distinct_refinement_block() {
        assert_eq!(guidance(1., false), (1., 0., 1., &[][..]));
        assert_eq!(guidance(0.99, false), (8., 4., 0.5, &[28][..]));
        assert_eq!(guidance(0.3, true), (1., 1., 1., &[27][..]));
    }
}
