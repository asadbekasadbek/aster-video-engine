//! Research feature extraction with the CoTracker3 offline checkpoint (CC BY-NC 4.0).
//! Architecture: Meta CoTracker3 BasicEncoder. No track coordinates are inferred here.
use crate::weights::{self, Result};
use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, fs, path::Path};

struct Encoder(HashMap<String, Tensor>);

fn norm(x: Tensor) -> candle_core::Result<Tensor> {
    let centered = x.broadcast_sub(&x.mean_keepdim((2, 3))?)?;
    centered.broadcast_div(&(centered.sqr()?.mean_keepdim((2, 3))? + 1e-5)?.sqrt()?)
}

impl Encoder {
    fn conv(&self, x: &Tensor, name: &str, pad: usize, stride: usize) -> Result<Tensor> {
        let weight = self
            .0
            .get(&format!("fnet.{name}.weight"))
            .ok_or_else(|| format!("missing {name} weight"))?;
        let bias = self
            .0
            .get(&format!("fnet.{name}.bias"))
            .ok_or_else(|| format!("missing {name} bias"))?;
        Ok(x.conv2d(weight, pad, stride, 1, 1)?
            .broadcast_add(&bias.reshape((1, (), 1, 1))?)?)
    }
    fn features(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let mut x = norm(self.conv(x, "conv1", 3, 2)?)?.relu()?;
        let mut levels = Vec::new();
        for layer in 1..=4 {
            for block in 0..2 {
                let prefix = format!("layer{layer}.{block}");
                let stride = if layer > 1 && block == 0 { 2 } else { 1 };
                let a = norm(self.conv(&x, &format!("{prefix}.conv1"), 1, stride)?)?.relu()?;
                let b = norm(self.conv(&a, &format!("{prefix}.conv2"), 1, 1)?)?.relu()?;
                if stride == 2 {
                    x = norm(self.conv(&x, &format!("{prefix}.downsample.0"), 0, 2)?)?;
                }
                x = (x + b)?.relu()?;
            }
            levels.push(x.upsample_bilinear2d(h / 4, w / 4, true)?);
        }
        let concat = Tensor::cat(&levels, 1)?;
        let projected = norm(self.conv(&concat, "conv2", 1, 1)?)?.relu()?;
        let desc = self.conv(&projected, "conv3", 0, 1)?;
        Ok(desc.broadcast_div(&desc.sqr()?.sum_keepdim(1)?.clamp(1e-12, f32::MAX)?.sqrt()?)?)
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let [model, photo, out] = args else {
        return Err("cotracker-features scaled_offline.pth FRAME NEW.safetensors".into());
    };
    weights::verify(
        Path::new(model),
        101890938,
        "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834",
    )?;
    let encoder = Encoder(candle_core::pickle::read_all(model)?.into_iter().collect());
    let rgb = image::open(photo)?.into_rgb8();
    let frame = Tensor::from_slice(
        rgb.as_raw(),
        (rgb.height() as usize, rgb.width() as usize, 3),
        &Device::Cpu,
    )?
    .to_dtype(DType::F32)?
    .permute((2, 0, 1))?
    .unsqueeze(0)?
    .contiguous()?
    .upsample_bilinear2d(384, 512, true)?;
    let desc = encoder.features(&(frame.affine(1. / 127.5, -1.)?))?;
    if desc.dims() != [1, 128, 96, 128]
        || desc
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .any(|x| !x.is_finite())
    {
        return Err("invalid features".into());
    }
    let bytes = safetensors::tensor::serialize([("fnet", desc)], None)?;
    use std::io::Write;
    fs::File::create_new(out)?.write_all(&bytes)?;
    println!("descriptor_shape=1x128x96x128 stride=4");
    Ok(())
}
