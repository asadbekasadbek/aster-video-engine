//! Two-output-teacher ablation for a small residual image processor, not a video generator.
use candle_core::{Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer};
use std::{collections::HashMap, path::Path};
#[path = "teacher_pilot/starter.rs"]
mod starter;
const W: usize = 576;
const H: usize = 384;
const F: usize = W * H * 3;
struct Clip {
    x: Vec<u8>,
    y: Vec<u8>,
}
fn load(d: &Path, name: &str) -> Result<Clip, Box<dyn std::error::Error>> {
    let read = |suffix: &str| -> std::io::Result<Vec<u8>> {
        let p = d.join(format!("{name}{suffix}.rgb"));
        if p.metadata()?.len() != 25 * F as u64 {
            return Err(std::io::Error::other("unexpected clip size"));
        }
        std::fs::read(p)
    };
    Ok(Clip {
        x: read("")?,
        y: read("-target")?,
    })
}
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> usize {
        let mut v = self.0;
        v ^= v << 13;
        v ^= v >> 17;
        v ^= v << 5;
        self.0 = v;
        v as usize
    }
}
// Shared channel filter: center intensity and eight local contrasts.
fn patch(v: &[u8], i: usize) -> [f32; 9] {
    let k = i % F / 3;
    let (y, x) = (k / W, k % W);
    let c = v[i] as f32;
    let mut out = [0.; 9];
    let mut n = 0;
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            let j = (i / F) * F
                + 3 * (y.saturating_add_signed(dy).min(H - 1) * W
                    + x.saturating_add_signed(dx).min(W - 1))
                + i % 3;
            out[n] = if n == 4 {
                (c - 128.) / 128.
            } else {
                (v[j] as f32 - c) / 32.
            };
            n += 1;
        }
    }
    out
}
struct Batch {
    inputs: Tensor,
    target: Tensor,
}
// Match a 5x5 neighborhood in green-channel intensity, with a conservative
// fallback for flat, ambiguous or poorly matching regions and image borders.
fn corresponding_pixel(current: &[u8], next: &[u8], p: usize) -> usize {
    let pixel = p / 3;
    let (x, y) = (pixel % W, pixel / W);
    if !(5..W - 5).contains(&x) || !(5..H - 5).contains(&y) {
        return p;
    }
    let center = pixel * 3 + 1;
    let mut best = (u32::MAX, p);
    let mut second = u32::MAX;
    for dy in -3isize..=3 {
        for dx in -3isize..=3 {
            let offset = 3 * (dy * W as isize + dx);
            let mut cost = 0;
            for sy in [-2isize, 0, 2] {
                for sx in [-2isize, 0, 2] {
                    let i = center.wrapping_add_signed(3 * (sy * W as isize + sx));
                    cost += u32::from(current[i].abs_diff(next[i.wrapping_add_signed(offset)]));
                }
            }
            if cost < best.0 {
                second = best.0;
                best = (cost, p.wrapping_add_signed(offset));
            } else {
                second = second.min(cost);
            }
        }
    }
    if best.0 <= 72 && second - best.0 >= 6 {
        best.1
    } else {
        p
    }
}

fn batch(clips: &[Clip], r: &mut Rng, n: usize, tracked: bool) -> candle_core::Result<Batch> {
    let mut inputs = vec![0.; 2 * n * 9];
    let mut target = vec![0.; 2 * n];
    for row in 0..n {
        let clip = &clips[r.next() % clips.len()];
        let t = r.next() % 24;
        let p = r.next() % F;
        let next = if tracked {
            corresponding_pixel(
                &clip.x[t * F..(t + 1) * F],
                &clip.x[(t + 1) * F..(t + 2) * F],
                p,
            )
        } else {
            p
        };
        for (k, position) in [p, next].into_iter().enumerate() {
            let i = (t + k) * F + position;
            let j = k * n + row;
            inputs[9 * j..9 * j + 9].copy_from_slice(&patch(&clip.x, i));
            target[j] = (clip.y[i] as f32 - clip.x[i] as f32) / 32.;
        }
    }
    Ok(Batch {
        inputs: Tensor::from_vec(inputs, (2 * n, 9), &Device::Cpu)?,
        target: Tensor::from_vec(target, (2 * n, 1), &Device::Cpu)?,
    })
}
struct Net {
    a: Var,
    b: Var,
    c: Var,
}
impl Net {
    fn new() -> candle_core::Result<Self> {
        let mut r = Rng(2026);
        let a = (0..108)
            .map(|_| ((r.next() % 1001) as f32 / 500. - 1.) * 0.4)
            .collect::<Vec<_>>();
        Ok(Self {
            a: Var::from_tensor(&Tensor::from_vec(a, (9, 12), &Device::Cpu)?)?,
            b: Var::from_tensor(&Tensor::full(0.05f32, (1, 12), &Device::Cpu)?)?,
            c: Var::zeros((12, 1), candle_core::DType::F32, &Device::Cpu)?,
        })
    }
    fn predict(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x.matmul(&self.a)?
            .broadcast_add(&self.b)?
            .relu()?
            .matmul(&self.c)
    }
}
fn losses(pred: &Tensor, b: &Batch) -> candle_core::Result<(Tensor, Tensor)> {
    let n = pred.dim(0)? / 2;
    Ok((
        (pred - &b.target)?.sqr()?.mean_all()?,
        (pred.narrow(0, n, n)? - pred.narrow(0, 0, n)?)?
            .sqr()?
            .mean_all()?,
    ))
}
fn train(clips: &[Clip], lambda: f64, tracked: bool) -> candle_core::Result<Net> {
    let net = Net::new()?;
    let mut r = Rng(2027);
    let mut opt = AdamW::new_lr(vec![net.a.clone(), net.b.clone(), net.c.clone()], 0.004)?;
    for _ in 0..600 {
        let b = batch(clips, &mut r, 512, tracked)?;
        let (s, t) = losses(&net.predict(&b.inputs)?, &b)?;
        opt.backward_step(&(s + t.affine(lambda, 0.)?)?)?;
    }
    Ok(net)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["starter"] {
        return starter::run();
    }
    if let [cmd, checkpoint, input, output] = args.as_slice()
        && cmd == "apply"
    {
        return starter::apply(Path::new(checkpoint), Path::new(input), Path::new(output));
    }
    if args.is_empty() {
        let exe = std::env::current_exe()?;
        let root = exe.parent().ok_or("missing executable directory")?;
        return trial(&root.join("data"), root);
    }
    if let [cmd, data, out] = args.as_slice()
        && cmd == "train"
    {
        return trial(Path::new(data), Path::new(out));
    }
    if let [cmd, checkpoint, rgb, out, ffmpeg] = args.as_slice()
        && cmd == "render"
    {
        let raw = std::fs::read(rgb)?;
        if raw.len() != 25 * F {
            return Err("expected 25 RGB24 frames at 576x384".into());
        }
        let net = load_net(Path::new(checkpoint))?;
        return render(&net, &raw, Path::new(out), Path::new(ffmpeg));
    }
    let [directory, ffmpeg] = args.as_slice() else {
        return Err("teacher_pilot DATA_DIR FFMPEG".into());
    };
    let d = Path::new(directory);
    let checkpoint = d.join("student.safetensors");
    let video = d.join("student.mp4");
    if checkpoint.exists() || video.exists() {
        return Err("refusing to overwrite trial".into());
    }
    let data = [load(d, "train-a")?, load(d, "train-b")?];
    let spatial = train(&data, 0., false)?;
    let both = train(&data, 1., false)?;
    let hold = load(d, "holdout")?;
    let b = batch(std::slice::from_ref(&hold), &mut Rng(2028), 65536, false)?;
    for (name, p) in [
        ("identity", b.target.zeros_like()?),
        ("sr_teacher", b.target.clone()),
        ("detail_only", spatial.predict(&b.inputs)?),
        ("two_signals", both.predict(&b.inputs)?),
    ] {
        let (s, t) = losses(&p, &b)?;
        println!(
            "{name} detail_target_rmse={:.5} adjacent_delta_rmse={:.5} units=RGB8",
            32. * s.to_scalar::<f32>()?.sqrt(),
            32. * t.to_scalar::<f32>()?.sqrt()
        );
    }
    save_net(&both, &checkpoint)?;
    render(&both, &hold.x, &video, Path::new(ffmpeg))
}

fn save_net(both: &Net, checkpoint: &Path) -> candle_core::Result<()> {
    let weights = HashMap::from([
        ("a", both.a.as_tensor().clone()),
        ("b", both.b.as_tensor().clone()),
        ("c", both.c.as_tensor().clone()),
    ]);
    candle_core::safetensors::save(&weights, checkpoint)
}

fn load_net(path: &Path) -> candle_core::Result<Net> {
    let tensors = candle_core::safetensors::load(path, &Device::Cpu)?;
    let get = |name: &str| {
        Var::from_tensor(
            tensors
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing {name}")))?,
        )
    };
    Ok(Net {
        a: get("a")?,
        b: get("b")?,
        c: get("c")?,
    })
}

fn trial(data: &Path, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let baseline = load_net(&data.join("student.safetensors"))?;
    let clips = [load(data, "train-a")?, load(data, "train-b")?];
    let hold = [load(data, "holdout")?];
    println!("Training on CPU. The reference model is preserved.");
    let candidate = train(&clips, 1., true)?;
    let mut report = String::from("model,pairing,detail_rmse,temporal_rmse\n");
    for (tracked, pairing) in [(false, "fixed"), (true, "tracked")] {
        let b = batch(&hold, &mut Rng(2028), 65536, tracked)?;
        for (name, net) in [("baseline", &baseline), ("candidate", &candidate)] {
            let (s, t) = losses(&net.predict(&b.inputs)?, &b)?;
            report.push_str(&format!(
                "{name},{pairing},{:.5},{:.5}\n",
                32. * s.to_scalar::<f32>()?.sqrt(),
                32. * t.to_scalar::<f32>()?.sqrt()
            ));
        }
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    let run = out.join(format!("run-{stamp}"));
    std::fs::create_dir(&run)?;
    save_net(&candidate, &run.join("candidate.safetensors"))?;
    std::fs::write(run.join("metrics.csv"), &report)?;
    println!("{report}{}", run.display());
    Ok(())
}

fn render(
    both: &Net,
    raw: &[u8],
    video: &Path,
    ffmpeg: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = aster_core::video::VideoSpec {
        width: W,
        height: H,
        frames: 25,
        fps_num: 24,
        fps_den: 1,
    };
    aster_core::mp4::encode(ffmpeg, video, spec, |w| {
        for (t, frame) in raw.as_chunks::<F>().0.iter().enumerate() {
            let mut bytes = Vec::with_capacity(F);
            for off in (0..F).step_by(8192) {
                let end = (off + 8192).min(F);
                let features = (off..end).flat_map(|i| patch(frame, i)).collect::<Vec<_>>();
                let run = || -> candle_core::Result<Vec<f32>> {
                    both.predict(&Tensor::from_vec(features, (end - off, 9), &Device::Cpu)?)?
                        .flatten_all()?
                        .to_vec1()
                };
                for (i, r) in run()
                    .map_err(std::io::Error::other)?
                    .into_iter()
                    .enumerate()
                {
                    if !r.is_finite() {
                        return Err(std::io::Error::other("nonfinite residual"));
                    }
                    bytes.push((frame[off + i] as f32 + 32. * r).clamp(0., 255.).round() as u8);
                }
            }
            w.write_frame(t, &bytes)?;
        }
        Ok(())
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_translated_texture_and_rejects_ambiguity() {
        let mut rng = Rng(77);
        let image: Vec<u8> = (0..F).map(|_| rng.next() as u8).collect();
        let shift = 3 * (W + 2);
        let mut next = vec![0; F];
        next[shift..].copy_from_slice(&image[..F - shift]);
        let p = 3 * (20 * W + 30) + 2;
        assert_eq!(corresponding_pixel(&image, &next, p), p + shift);
        next.fill(100);
        assert_eq!(corresponding_pixel(&next, &next, p), p);
        assert_eq!(corresponding_pixel(&image, &next, 0), 0);
    }

    #[test]
    fn temporal_loss_compares_matching_pairs() -> candle_core::Result<()> {
        let pred = Tensor::from_vec(vec![0f32, 2., 1., 5.], (4, 1), &Device::Cpu)?;
        let batch = Batch {
            inputs: pred.clone(),
            target: pred.clone(),
        };
        let (spatial, temporal) = losses(&pred, &batch)?;
        assert_eq!(spatial.to_scalar::<f32>()?, 0.);
        assert_eq!(temporal.to_scalar::<f32>()?, 5.);
        Ok(())
    }
}
