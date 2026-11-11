//! Research-only inference for one queried point across two frames, on the
//! CoTracker3 512x384 working grid. Meta checkpoint is licensed CC BY-NC 4.0.
use crate::weights::{self, Result};
use candle_core::{D, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use std::path::Path;

fn normalized(x: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
    centered.broadcast_div(&(centered.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?)
}

fn linear(x: &Tensor, v: VarBuilder, out: usize) -> candle_core::Result<Tensor> {
    candle_nn::linear(x.dim(D::Minus1)?, out, v)?.forward(x)
}
fn block(x: &Tensor, context: Option<&Tensor>, v: VarBuilder) -> candle_core::Result<Tensor> {
    let q_input = normalized(x, 1e-6)?;
    let (ctx, att) = match context {
        Some(c) => (
            candle_nn::layer_norm(384, 1e-5, v.pp("norm_context"))?.forward(c)?,
            v.pp("cross_attn"),
        ),
        None => (q_input.clone(), v.pp("attn")),
    };
    let heads = |t: Tensor| -> candle_core::Result<Tensor> {
        let (batch, seq, _) = t.dims3()?;
        t.reshape((batch, seq, 8, 48))?
            .transpose(1, 2)?
            .contiguous()
    };
    let q = heads(linear(&q_input, att.pp("to_q"), 384)?)?;
    let kv = linear(&ctx, att.pp("to_kv"), 768)?;
    let k = heads(kv.narrow(2, 0, 384)?)?;
    let value = heads(kv.narrow(2, 384, 384)?)?;
    let scores = q
        .matmul(&k.transpose(2, 3)?.contiguous()?)?
        .affine(1. / 48f64.sqrt(), 0.)?;
    let mixed = candle_nn::ops::softmax_last_dim(&scores)?
        .matmul(&value)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape(x.shape())?;
    let x = (x + linear(&mixed, att.pp("to_out"), 384)?)?;
    let hidden = linear(&normalized(&x, 1e-6)?, v.pp("mlp.fc1"), 1536)?.gelu()?;
    x + linear(&hidden, v.pp("mlp.fc2"), 384)?
}

fn update(input: &Tensor, v: VarBuilder) -> candle_core::Result<Tensor> {
    let virtuals = v
        .get((1, 64, 1, 384), "virual_tracks")?
        .reshape((64, 1, 384))?
        .repeat((1, 2, 1))?;
    let mut tokens = Tensor::cat(&[linear(input, v.pp("input_transform"), 384)?, virtuals], 0)?;
    for i in 0..3 {
        let temporal = block(&tokens, None, v.pp(format!("time_blocks.{i}")))?
            .transpose(0, 1)?
            .contiguous()?;
        let point = temporal.narrow(1, 0, 1)?;
        let virtuals = block(
            &temporal.narrow(1, 1, 64)?,
            Some(&point),
            v.pp(format!("space_virtual2point_blocks.{i}")),
        )?;
        let virtuals = block(&virtuals, None, v.pp(format!("space_virtual_blocks.{i}")))?;
        let point = block(
            &point,
            Some(&virtuals),
            v.pp(format!("space_point2virtual_blocks.{i}")),
        )?;
        tokens = Tensor::cat(&[point, virtuals], 1)?
            .transpose(0, 1)?
            .contiguous()?;
    }
    let point = tokens.narrow(0, 0, 1)?;
    Tensor::cat(
        &[
            linear(&point, v.pp("flow_head"), 2)?,
            linear(&point, v.pp("vis_conf_head"), 2)?,
        ],
        2,
    )?
    .squeeze(0)
}

struct Map {
    data: Vec<f32>,
    w: usize,
    h: usize,
}
impl Map {
    fn neighborhood(&self, xy: [f32; 2]) -> candle_core::Result<Tensor> {
        let mut values = Vec::with_capacity(49 * 128);
        for dx in -3..=3 {
            for dy in -3..=3 {
                let x = (xy[0] + dx as f32).clamp(0., (self.w - 1) as f32);
                let y = (xy[1] + dy as f32).clamp(0., (self.h - 1) as f32);
                let (ix, iy) = (x as usize, y as usize);
                let (fx, fy) = (x.fract(), y.fract());
                for c in 0..128 {
                    let at = |row: usize, col: usize| {
                        self.data[(row.min(self.h - 1) * self.w + col.min(self.w - 1)) * 128 + c]
                    };
                    values.push(
                        (1. - fy) * ((1. - fx) * at(iy, ix) + fx * at(iy, ix + 1))
                            + fy * ((1. - fx) * at(iy + 1, ix) + fx * at(iy + 1, ix + 1)),
                    );
                }
            }
        }
        Tensor::from_vec(values, (49, 128), &Device::Cpu)
    }
}
fn pyramid(path: &str) -> Result<Vec<Map>> {
    let mut t = candle_core::safetensors::load(path, &Device::Cpu)?
        .remove("fnet")
        .ok_or("missing fnet")?;
    if t.dims() != [1, 128, 96, 128] {
        return Err("invalid CoTracker feature map".into());
    }
    let mut out = Vec::new();
    for i in 0..4 {
        let (_, _, h, w) = t.dims4()?;
        let data = t
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if data.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite source features".into());
        }
        out.push(Map { data, w, h });
        if i < 3 {
            t = t.avg_pool2d(2)?;
        }
    }
    Ok(out)
}
fn position(motion: [f32; 4]) -> Vec<f32> {
    let mut data = motion.to_vec();
    for phase in [0., std::f32::consts::FRAC_PI_2] {
        for k in 0..10 {
            for v in motion {
                data.push((v * ((1 << k) as f32) + phase).sin());
            }
        }
    }
    data
}

pub fn run(args: &[String]) -> Result<()> {
    let [model, first, second, x, y] = args else {
        return Err(
            "cotracker-pair MODEL FIRST_FEATURES SECOND_FEATURES X Y (512x384 coordinates)".into(),
        );
    };
    let query = [x.parse::<f32>()?, y.parse()?];
    if !(0. ..=511.).contains(&query[0]) || !(0. ..=383.).contains(&query[1]) {
        return Err("point outside model grid".into());
    }
    weights::verify(
        Path::new(model),
        101890938,
        "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834",
    )?;
    let tensors = candle_core::pickle::read_all(model)?
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    let time = tensors.get("time_emb").ok_or("missing time_emb")?;
    let (_, window, _) = time.dims3()?;
    let time = time
        .transpose(1, 2)?
        .unsqueeze(2)?
        .contiguous()?
        .upsample_bilinear2d(1, 2, false)?
        .squeeze(2)?
        .transpose(1, 2)?;
    if window < 2 {
        return Err("invalid tracker time window".into());
    }
    let v = VarBuilder::from_tensors(tensors, candle_core::DType::F32, &Device::Cpu);
    let maps = [pyramid(first)?, pyramid(second)?];
    let mut xy = [query.map(|v| v / 4.); 2];
    let mut vc = [[0f32; 2]; 2];
    let mut support = Vec::new();
    for (level, map) in maps[0].iter().enumerate() {
        support.push(
            map.neighborhood(xy[0].map(|x| x / (1 << level) as f32))?
                .t()?
                .contiguous()?,
        );
    }
    for _ in 0..6 {
        let mut input = Vec::new();
        let motion = [(xy[0][0] - xy[1][0]) / 128., (xy[0][1] - xy[1][1]) / 96.];
        for frame in 0..2 {
            input.extend(vc[frame]);
            for (level, reference) in support.iter().enumerate() {
                let patch =
                    maps[frame][level].neighborhood(xy[frame].map(|x| x / (1 << level) as f32))?;
                let corr = patch.matmul(reference)?.reshape((1, 2401))?;
                let hidden = linear(&corr, v.pp("corr_mlp.fc1"), 384)?.gelu_erf()?;
                input.extend(
                    linear(&hidden, v.pp("corr_mlp.fc2"), 256)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                );
            }
            input.extend(position(if frame == 0 {
                [motion[0], motion[1], 0., 0.]
            } else {
                [0., 0., -motion[0], -motion[1]]
            }));
        }
        let input = (Tensor::from_vec(input, (1, 2, 1110), &Device::Cpu)? + &time)?;
        let delta = update(&input, v.pp("updateformer"))?.to_vec2::<f32>()?;
        for t in 0..2 {
            for k in 0..2 {
                xy[t][k] += delta[t][k];
                vc[t][k] += delta[t][2 + k];
            }
        }
        if xy.iter().chain(&vc).flatten().any(|v| !v.is_finite()) {
            return Err("nonfinite tracking update".into());
        }
    }
    let prob = |x: f32| 1. / (1. + (-x).exp());
    for t in 0..2 {
        println!(
            "frame={t} x={:.3} y={:.3} visibility={:.5} confidence={:.5}",
            xy[t][0] * 4.,
            xy[t][1] * 4.,
            prob(vc[t][0]),
            prob(vc[t][1])
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_grid_starts_with_x_offset_then_advances_y_and_clamps_edges() {
        let map = Map {
            w: 2,
            h: 2,
            data: (0..4)
                .flat_map(|i| std::iter::repeat_n(i as f32, 128))
                .collect(),
        };
        let patch = map
            .neighborhood([0.5, 0.5])
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(patch[0][0], 0.);
        assert_eq!(patch[3][0], 1.);
        assert_eq!(patch[24][0], 1.5);
        assert_eq!(patch[48][0], 3.);
    }
}
