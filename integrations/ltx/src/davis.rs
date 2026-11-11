//! Region overlap for one DAVIS 2017 semi-supervised validation sequence.
use crate::weights::Result;
use std::{collections::BTreeSet, fs, io::BufReader, path::Path};

struct Mask {
    size: (u32, u32),
    labels: Vec<u8>,
}

fn read(path: &Path) -> Result<Mask> {
    let decoder = png::Decoder::new(BufReader::new(fs::File::open(path)?));
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    if !matches!(
        info.color_type,
        png::ColorType::Indexed | png::ColorType::Grayscale
    ) || info.bit_depth != png::BitDepth::Eight
    {
        return Err("expected 8-bit indexed or grayscale object IDs".into());
    }
    let size = (info.width, info.height);
    let n = (size.0 as usize)
        .checked_mul(size.1 as usize)
        .filter(|n| *n <= 16_777_216)
        .ok_or("mask too large")?;
    let mut labels = vec![0; n];
    reader.next_frame(&mut labels)?;
    Ok(Mask { size, labels })
}

fn overlap(gt: &[u8], prediction: &[u8], id: u8) -> f64 {
    let (mut intersection, mut union) = (0u64, 0u64);
    for (&g, &p) in gt.iter().zip(prediction) {
        if g == 255 {
            continue;
        }
        intersection += u64::from(g == id && p == id);
        union += u64::from(g == id || p == id);
    }
    if union == 0 {
        1.
    } else {
        intersection as f64 / union as f64
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let [root, sequence, predictions] = args else {
        return Err("davis-iou DAVIS_ROOT VAL_SEQUENCE PREDICTION_SEQUENCE_DIR".into());
    };
    let root = Path::new(root);
    if sequence.is_empty()
        || !sequence
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
        || !fs::read_to_string(root.join("ImageSets/2017/val.txt"))?
            .lines()
            .any(|s| s.trim() == sequence)
    {
        return Err("sequence is outside the DAVIS 2017 validation split".into());
    }
    let mut frames = Vec::new();
    for entry in fs::read_dir(root.join("Annotations/480p").join(sequence))? {
        let p = entry?.path();
        if p.extension().is_some_and(|e| e == "png") {
            frames.push(p);
        }
    }
    frames.sort();
    if frames.len() < 3 {
        return Err("incomplete DAVIS sequence".into());
    }
    let first = read(&frames[0])?;
    let ids: BTreeSet<_> = first
        .labels
        .iter()
        .copied()
        .filter(|&id| id != 0 && id != 255)
        .collect();
    if ids.is_empty() {
        return Err("first frame does not identify any objects".into());
    }
    let mut sum = 0.;
    for p in &frames[1..frames.len() - 1] {
        let gt = read(p)?;
        let pred = read(&Path::new(predictions).join(p.file_name().ok_or("invalid frame path")?))?;
        if pred.size != gt.size || gt.size != first.size {
            return Err("mask dimensions differ".into());
        }
        if gt
            .labels
            .iter()
            .any(|id| !matches!(id, 0 | 255) && !ids.contains(id))
        {
            return Err("unexpected ground-truth object".into());
        }
        for &id in &ids {
            sum += overlap(&gt.labels, &pred.labels, id);
        }
    }
    let n = frames.len() - 2;
    println!(
        "sequence={sequence} evaluated_frames={n} objects={} mean_object_frame_iou={:.6}",
        ids.len(),
        sum / (ids.len() * n) as f64
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_ids_ignores_void_and_scores_absence() {
        assert_eq!(overlap(&[1, 1, 2, 255], &[1, 2, 1, 1], 1), 1. / 3.);
        assert_eq!(overlap(&[0, 255], &[0, 1], 1), 1.);
        assert_eq!(overlap(&[0, 2], &[1, 2], 1), 0.);
    }
}
