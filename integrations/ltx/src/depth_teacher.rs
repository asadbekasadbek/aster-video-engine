use crate::weights::{self, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config},
    dinov2,
};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, io::Write, path::Path, sync::Arc};

const SIDE: usize = 518;
const DIGEST: &str = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378";

pub fn run(args: &[String]) -> Result<()> {
    let [model, image_path, output] = args else {
        return Err("depth-frame MODEL.pth FRAME.jpg NEW.safetensors".into());
    };
    if Path::new(output).exists() {
        return Err("depth output exists".into());
    }
    weights::verify(Path::new(model), 99_218_434, DIGEST)?;
    // The pinned CUDA backend lacks DINO's layer-norm kernel.
    let dev = Device::Cpu;
    let mut tensors: HashMap<_, _> = candle_core::pickle::read_all(model)?.into_iter().collect();
    // Candle's DINO constructor also constructs an ImageNet classifier. DepthAnything
    // only calls get_intermediate_layers; the released depth checkpoint has no classifier.
    for (key, shape) in [
        ("pretrained.head.weight", vec![1000, 768]),
        ("pretrained.head.bias", vec![1000]),
    ] {
        tensors
            .entry(key.to_owned())
            .or_insert(Tensor::zeros(shape, DType::F32, &Device::Cpu)?);
    }
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
    let backbone = dinov2::vit_small(vb.pp("pretrained"))?;
    let net = DepthAnythingV2::new(Arc::new(backbone), DepthAnythingV2Config::vit_small(), vb)?;
    let source = fs::read(image_path)?;
    let rgb = image::load_from_memory(&source)?.into_rgb8();
    let (w, h) = rgb.dimensions();
    let crop = w.min(h);
    let (x, y) = ((w - crop) / 2, (h - crop) / 2);
    let patch = image::imageops::crop_imm(&rgb, x, y, crop, crop);
    let resized = image::imageops::resize(
        &*patch,
        SIDE as u32,
        SIDE as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let pixel = Tensor::from_slice(resized.as_raw(), (SIDE, SIDE, 3), &dev)?
        .to_dtype(DType::F32)?
        .affine(1. / 255., 0.)?;
    let mean = Tensor::new(&[0.485f32, 0.456, 0.406], &dev)?;
    let std = Tensor::new(&[0.229f32, 0.224, 0.225], &dev)?;
    let input = pixel
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?
        .permute((2, 0, 1))?
        .contiguous()?
        .unsqueeze(0)?;
    // The pinned DPT head pads its final 1x1 convolution by one pixel.
    // Its interior is exactly the unpadded output of that convolution.
    let native = net.forward(&input)?;
    if native.dims() != [1, 1, SIDE + 2, SIDE + 2] {
        return Err("unexpected DPT padded output".into());
    }
    let prediction = native
        .narrow(2, 1, SIDE)?
        .narrow(3, 1, SIDE)?
        .contiguous()?;
    if prediction.dims() != [1, 1, SIDE, SIDE] {
        return Err(format!("unexpected depth shape {:?}", prediction.dims()).into());
    }
    let values = prediction.flatten_all()?.to_vec1::<f32>()?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite depth target".into());
    }
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![SIDE, SIDE], &bytes)?;
    let metadata = HashMap::from([
        ("model_sha256".into(), DIGEST.to_owned()),
        (
            "frame_sha256".into(),
            format!("{:x}", Sha256::digest(source)),
        ),
        ("source_crop_xywh".into(), format!("{x},{y},{crop},{crop}")),
        ("source_wh".into(), format!("{w},{h}")),
    ]);
    let serialized = safetensors::tensor::serialize([("relative_depth", view)], Some(metadata))?;
    fs::File::create_new(output)?.write_all(&serialized)?;
    println!(
        "device=cpu relative_depth={}x{} range={:?}..{:?} output={output}",
        SIDE,
        SIDE,
        values.iter().copied().reduce(f32::min),
        values.iter().copied().reduce(f32::max)
    );
    Ok(())
}
