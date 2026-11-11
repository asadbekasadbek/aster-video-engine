use crate::weights::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3_vl::{Config, Qwen3VLModel};
use std::{fs, path::Path};
use tokenizers::Tokenizer;

fn pack_image(rgb: &image::RgbImage) -> Vec<f32> {
    let (w, h) = rgb.dimensions();
    let mut pixels = Vec::new();
    for by in (0..h).step_by(32) {
        for bx in (0..w).step_by(32) {
            for py in [0, 16] {
                for px in [0, 16] {
                    for c in 0..3 {
                        for _ in 0..2 {
                            for y in 0..16 {
                                for x in 0..16 {
                                    pixels.push(
                                        f32::from(rgb.get_pixel(bx + px + x, by + py + y)[c])
                                            / 127.5
                                            - 1.,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    pixels
}

pub fn run(args: &[String]) -> Result<()> {
    let (args, cuda) = match args {
        [rest @ .., flag] if flag == "--cuda" => (rest, true),
        _ => (args, false),
    };
    let [model_dir, photo] = args else {
        return Err("qwen-caption MODEL_DIRECTORY FRAME [--cuda]".into());
    };
    let dir = Path::new(model_dir);
    let (device, dtype) = if cuda {
        if !cfg!(feature = "teacher-cuda") {
            return Err("rebuild with --features teacher-cuda".into());
        }
        let status = std::process::Command::new("nvidia-smi")
            .args([
                "-i",
                "0",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output()?;
        if !status.status.success()
            || std::str::from_utf8(&status.stdout)?.trim().parse::<u64>()? < 10752
        {
            return Err(
                "Qwen BF16 requires at least 10752 MiB free on GPU 0 for this configuration".into(),
            );
        }
        (Device::new_cuda(0)?, DType::BF16)
    } else {
        (Device::Cpu, DType::F32)
    };
    let mut config: Config = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
    config.text_config.max_position_embeddings = 256;
    let shards = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    .map(|s| dir.join(s));
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    // Safety: the two completed local checkpoint files must remain immutable during inference.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, &device)? };
    eprintln!("qwen: loading {dtype:?} model");
    let model = Qwen3VLModel::new(&config, vb)?;
    let rgb = image::open(photo)?.into_rgb8();
    let longest = rgb.width().max(rgb.height()) as f32;
    let extent = |n: u32| ((n as f32 / longest * 8.).round() as u32).max(1) * 32;
    let (w, h) = (extent(rgb.width()), extent(rgb.height()));
    let rgb = image::imageops::resize(&rgb, w, h, image::imageops::FilterType::CatmullRom);
    let pixels = pack_image(&rgb);
    let rows = (w * h / 256) as usize;
    let grid = Tensor::new(&[[1u32, h / 16, w / 16]], &device)?;
    let vision = Tensor::from_vec(pixels, (rows, 1536), &device)?.to_dtype(dtype)?;
    let encode = |s: &str| {
        tokenizer
            .encode(s, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string())
    };
    let prefix = encode("<|im_start|>user\n<|vision_start|>")?;
    let suffix = encode(
        "<|vision_end|>Describe the main subject and its surroundings in one short sentence.<|im_end|>\n<|im_start|>assistant\n",
    )?;
    let mut pos = 0;
    let token_step = |id: u32, pos: usize| -> candle_core::Result<Tensor> {
        model.forward(
            &Tensor::new(&[[id]], &device)?,
            None,
            None,
            None,
            None,
            vec![1],
            vec![vec![]],
            vec![vec![]],
            &[pos],
        )
    };
    eprintln!("qwen: processing prompt and image");
    for id in prefix {
        token_step(id, pos)?;
        pos += 1;
    }
    let n = rows / 4;
    let pad = Tensor::from_vec(vec![151655u32; n], (1, n), &device)?;
    model.forward(
        &pad,
        Some(vision),
        None,
        Some(grid),
        None,
        vec![n],
        vec![vec![(0, n)]],
        vec![vec![]],
        &[pos],
    )?;
    pos += n;
    let mut logits = None;
    for id in suffix {
        logits = Some(token_step(id, pos)?);
        pos += 1;
    }
    let mut logits = logits.ok_or("empty assistant prefix")?;
    let mut ids = Vec::new();
    eprintln!("qwen: decoding caption");
    for step in 0..40 {
        let id = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
        if matches!(id, 151643 | 151645) {
            break;
        }
        ids.push(id);
        if (step + 1) % 8 == 0 {
            eprintln!("qwen: decoded {} tokens", step + 1);
        }
        logits = token_step(id, pos)?;
        pos += 1;
    }
    if ids.is_empty() {
        return Err("empty Qwen caption".into());
    }
    println!(
        "{}",
        tokenizer.decode(&ids, true).map_err(|e| e.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn groups_four_patches_before_advancing_to_next_merged_column() {
        let rgb = image::RgbImage::from_fn(64, 32, |x, y| image::Rgb([x as u8, y as u8, 255]));
        let values = pack_image(&rgb);
        let unscale = |v: f32| ((v + 1.) * 127.5).round() as u8;
        assert_eq!(values.len(), 8 * 1536);
        assert_eq!(unscale(values[1536]), 16);
        assert_eq!(unscale(values[2 * 1536 + 512]), 16);
        assert_eq!(unscale(values[4 * 1536]), 32);
        assert_eq!(values[15], values[256 + 15]);
    }
}
