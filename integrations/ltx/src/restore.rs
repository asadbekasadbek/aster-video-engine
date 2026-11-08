//! Offline SRVGG restoration trial for the existing 704x384, 72-frame RGB clip.
use crate::weights::{self, Result};
use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, fs::File, io::Read, path::Path, time::Instant};

struct Layer {
    kernel: Tensor,
    bias: Tensor,
    slope: Option<Tensor>,
}

fn infer(input: &Tensor, layers: &[Layer]) -> candle_core::Result<Tensor> {
    let mut x = input.clone();
    for l in layers {
        x = x.conv2d(&l.kernel, 1, 1, 1, 1)?.broadcast_add(&l.bias)?;
        if let Some(a) = &l.slope {
            x = (x.relu()? + x.minimum(0.)?.broadcast_mul(a)?)?;
        }
    }
    let (_, _, h, w) = input.dims4()?;
    let residual = candle_nn::ops::pixel_shuffle(&x, 4)?;
    (residual + input.upsample_nearest2d(h * 4, w * 4)?)?
        .to_dtype(DType::F32)?
        .avg_pool2d(2)
}

pub fn run(args: &[String]) -> Result<()> {
    let [model, raw, out, ffmpeg] = args else {
        return Err("restore-glance EXACT.pth INPUT.rgb NEW.mp4 FFMPEG".into());
    };
    weights::verify(
        Path::new(model),
        4_885_111,
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    )?;
    let mut file = File::open(raw)?;
    const H: usize = 384;
    const W: usize = 704;
    const N: usize = 72;
    if file.metadata()?.len() != (H * W * 3 * N) as u64 {
        return Err("expected 72 RGB24 frames at 704x384".into());
    }
    let _monitor = crate::telemetry::Monitor::start();
    let start = Instant::now();
    let device = Device::new_cuda(0)?;
    let layers = load_layers(model, &device)?;
    let spec = aster_core::video::VideoSpec {
        width: W * 2,
        height: H * 2,
        frames: N,
        fps_num: 24,
        fps_den: 1,
    };
    let mut bytes = vec![0; H * W * 3];
    aster_core::mp4::encode(Path::new(ffmpeg), Path::new(out), spec, |writer| {
        for i in 0..N {
            file.read_exact(&mut bytes)?;
            let generate = || -> Result<Vec<u8>> {
                let input = Tensor::from_slice(&bytes, (H, W, 3), &device)?
                    .to_dtype(DType::F16)?
                    .affine(1. / 255., 0.)?
                    .permute((2, 0, 1))?
                    .unsqueeze(0)?
                    .contiguous()?;
                let restored = infer(&input, &layers)?
                    .squeeze(0)?
                    .permute((1, 2, 0))?
                    .flatten_all()?
                    .affine(2., -1.)?
                    .to_vec1::<f32>()?;
                Ok(crate::inference::rgb_bytes(restored)?)
            };
            writer.write_frame(
                i,
                &generate().map_err(|e| std::io::Error::other(e.to_string()))?,
            )?;
            if i % 12 == 0 {
                println!(
                    "restored_frames={} elapsed_ms={}",
                    i + 1,
                    start.elapsed().as_millis()
                );
            }
        }
        Ok(())
    })?;
    println!(
        "restored=72 elapsed_ms={} peak_worker_bytes={:?}",
        start.elapsed().as_millis(),
        aster_core::memory::peak_working_set_bytes()?
    );
    Ok(())
}

fn load_layers(model: &str, device: &Device) -> Result<Vec<Layer>> {
    let tensors: HashMap<_, _> = candle_core::pickle::read_all_with_key(model, Some("params"))?
        .into_iter()
        .collect();
    let get = |name: String, shape: &[usize]| -> Result<Tensor> {
        let t = tensors
            .get(&name)
            .ok_or_else(|| format!("missing {name}"))?;
        if t.dims() != shape {
            return Err(format!("wrong tensor shape {name}").into());
        }
        Ok(t.to_dtype(DType::F16)?.to_device(device)?)
    };
    let mut layers = Vec::new();
    for i in 0..34 {
        let width = if i == 33 { 48 } else { 64 };
        let kernel = get(
            format!("body.{}.weight", 2 * i),
            &[width, if i == 0 { 3 } else { 64 }, 3, 3],
        )?;
        let bias = get(format!("body.{}.bias", 2 * i), &[width])?.reshape((1, width, 1, 1))?;
        let slope = if i == 33 {
            None
        } else {
            Some(get(format!("body.{}.weight", 2 * i + 1), &[64])?.reshape((1, 64, 1, 1))?)
        };
        layers.push(Layer {
            kernel,
            bias,
            slope,
        });
    }
    Ok(layers)
}
/// Lossless native-size teacher targets on 25-frame, 576x384 text-only clips.
pub fn prepare(args: &[String]) -> Result<()> {
    use std::io::Write;
    let [model, raw, out] = args else {
        return Err("teacher-target MODEL RGB24 NEW.rgb".into());
    };
    weights::verify(
        Path::new(model),
        4885111,
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    )?;
    let (h, w) = (384, 576);
    let mut f = File::open(raw)?;
    if f.metadata()?.len() != (25 * h * w * 3) as u64 {
        return Err("expected 25 RGB24 frames at 576x384".into());
    }
    let device = Device::new_cuda(0)?;
    let layers = load_layers(model, &device)?;
    let mut dst = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)?;
    let mut buf = vec![0; h * w * 3];
    for _ in 0..25 {
        f.read_exact(&mut buf)?;
        let x = Tensor::from_slice(&buf, (h, w, 3), &device)?
            .to_dtype(DType::F16)?
            .affine(1. / 255., 0.)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?;
        let y = infer(&x, &layers)?
            .avg_pool2d(2)?
            .squeeze(0)?
            .permute((1, 2, 0))?
            .flatten_all()?
            .affine(2., -1.)?
            .to_vec1::<f32>()?;
        dst.write_all(&crate::inference::rgb_bytes(y)?)?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffled_residual_uses_rgb_channels_and_two_by_two_area_reduction() -> candle_core::Result<()>
    {
        let d = &Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3.], (1, 3, 1, 1), d)?;
        let layer = Layer {
            kernel: Tensor::zeros((48, 3, 3, 3), DType::F32, d)?,
            bias: Tensor::from_vec((0..48).map(|v| v as f32).collect(), (1, 48, 1, 1), d)?,
            slope: None,
        };
        let result = infer(&x, &[layer])?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            result,
            [
                3.5, 5.5, 11.5, 13.5, 20.5, 22.5, 28.5, 30.5, 37.5, 39.5, 45.5, 47.5
            ]
        );
        Ok(())
    }
}
