use candle_core::{Device, Tensor};
use sam2_rs::{Sam2InteractiveVideoPredictor, Sam2Variant, VideoPredictorConfig};
use std::{collections::BTreeSet, fs, io::BufReader, path::Path};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [checkpoint, folder, initial, count, out] = args.as_slice() else {
        return Err(
            "aster-sam2 CHECKPOINT JPEG_DIRECTORY FIRST_ID_MASK.png FRAME_COUNT NEW_DIRECTORY"
                .into(),
        );
    };
    let count: usize = count.parse()?;
    if !(2..=128).contains(&count) {
        return Err("frame count must be 2..128".into());
    }
    let mut paths = Vec::new();
    for e in fs::read_dir(folder)? {
        let p = e?.path();
        if p.extension().is_some_and(|s| s == "jpg") {
            paths.push(p);
        }
    }
    paths.sort();
    if paths.len() < count {
        return Err("insufficient frames".into());
    }
    paths.truncate(count);
    let frames = paths
        .iter()
        .map(|p| Ok(image::open(p)?.into_rgb8()))
        .collect::<Result<Vec<_>>>()?;
    let refs = frames
        .iter()
        .map(|f| {
            (
                f.as_raw().as_slice(),
                f.width() as usize,
                f.height() as usize,
            )
        })
        .collect::<Vec<_>>();
    let (_, w, h) = refs[0];
    if refs.iter().any(|(_, a, b)| (*a, *b) != (w, h)) {
        return Err("frame size changed".into());
    }
    let mut r = png::Decoder::new(BufReader::new(fs::File::open(initial)?)).read_info()?;
    let i = r.info();
    if (i.width as usize, i.height as usize) != (w, h)
        || i.bit_depth != png::BitDepth::Eight
        || !matches!(
            i.color_type,
            png::ColorType::Grayscale | png::ColorType::Indexed
        )
    {
        return Err("invalid first-frame ID mask".into());
    }
    let mut labels = vec![0; w * h];
    r.next_frame(&mut labels)?;
    let ids = labels
        .iter()
        .copied()
        .filter(|v| !matches!(v, 0 | 255))
        .collect::<BTreeSet<_>>();
    if ids.is_empty() {
        return Err("no objects in initial mask".into());
    }
    if Path::new(out).exists() {
        return Err("output directory exists".into());
    }
    let net = Sam2InteractiveVideoPredictor::load_sam2_1(
        Sam2Variant::HieraTiny,
        checkpoint,
        VideoPredictorConfig::default(),
    )?;
    let mut session = net.init_state(&refs)?;
    for id in ids {
        let mask = Tensor::from_vec(
            labels.iter().map(|p| f32::from(*p == id)).collect(),
            (h, w),
            &Device::Cpu,
        )?;
        net.add_new_mask(&mut session, 0, u32::from(id), &mask)?;
    }
    let outputs = net.propagate_in_video(&mut session, None, None, false)?;
    if outputs.len() != count {
        return Err("incomplete propagation".into());
    }
    fs::create_dir(out)?;
    for result in outputs {
        let n = result.object_ids.len();
        let scores = result.masks.reshape((n, h * w))?.to_vec2::<f32>()?;
        let mut mask = vec![0u8; h * w];
        let mut best = vec![0f32; h * w];
        for (row, id) in scores.iter().zip(result.object_ids) {
            for (j, &score) in row.iter().enumerate() {
                if !score.is_finite() {
                    return Err("nonfinite mask score".into());
                }
                if score > best[j] {
                    best[j] = score;
                    mask[j] = u8::try_from(id)?;
                }
            }
        }
        let mut output = Path::new(out).join(paths[result.frame_index].file_name().unwrap());
        output.set_extension("png");
        image::save_buffer(output, &mask, w as u32, h as u32, image::ColorType::L8)?;
    }
    println!("sam2_tracked_frames={count}");
    Ok(())
}
