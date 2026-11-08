//! Exact self attention in query slices: each query still sees all image tokens.
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_b, ops::softmax};
use candle_video::models::ltx_video::ltx_transformer::{RmsNorm, apply_rotary_emb};

#[derive(Clone, Debug)]
pub struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    nq: RmsNorm,
    nk: RmsNorm,
    heads: usize,
    dim: usize,
}
fn attend(q: &Tensor, k: &Tensor, v: &Tensor, chunk: usize) -> Result<Tensor> {
    let (_, _, len, dim) = q.dims4()?;
    let kt = k.transpose(2, 3)?;
    let scale = (1f32 / (dim as f32).sqrt()) as f64;
    let mut parts = Vec::new();
    for i in (0..len).step_by(chunk) {
        let scores = q
            .narrow(2, i, (len - i).min(chunk))?
            .contiguous()?
            .matmul(&kt)?
            .affine(scale, 0.)?;
        parts.push(softmax(&scores, D::Minus1)?.matmul(v)?);
    }
    Tensor::cat(&parts, 2)
}
impl SelfAttention {
    pub fn new(
        width: usize,
        heads: usize,
        dim: usize,
        bias: bool,
        out_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let n = heads * dim;
        Ok(Self {
            q: linear_b(width, n, bias, vb.pp("to_q"))?,
            k: linear_b(width, n, bias, vb.pp("to_k"))?,
            v: linear_b(width, n, bias, vb.pp("to_v"))?,
            out: linear_b(n, width, out_bias, vb.pp("to_out").pp("0"))?,
            nq: RmsNorm::new(n, 1e-5, true, vb.pp("norm_q"))?,
            nk: RmsNorm::new(n, 1e-5, true, vb.pp("norm_k"))?,
            heads,
            dim,
        })
    }
    pub fn forward(&self, x: &Tensor, rope: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let mut q = self.nq.forward(&self.q.forward(x)?)?;
        let mut k = self.nk.forward(&self.k.forward(x)?)?;
        if let Some((c, s)) = rope {
            q = apply_rotary_emb(&q, c, s)?;
            k = apply_rotary_emb(&k, c, s)?;
        }
        let split = |a: Tensor| {
            a.reshape((b, s, self.heads, self.dim))?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(DType::F32)
        };
        let y = attend(&split(q)?, &split(k)?, &split(self.v.forward(x)?)?, 512)?;
        self.out.forward(
            &y.to_dtype(x.dtype())?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, self.heads * self.dim))?,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uneven_slices_match_dense_across_heads() -> Result<()> {
        let q = Tensor::arange(0u32, 42, &candle_core::Device::Cpu)?
            .to_dtype(DType::F32)?
            .affine(0.05, -1.)?
            .reshape((1, 2, 7, 3))?;
        let k = q.sqr()?;
        let v = q.sin()?;
        let dense = softmax(
            &q.matmul(&k.transpose(2, 3)?)?
                .affine(1. / 3f64.sqrt(), 0.)?,
            D::Minus1,
        )?
        .matmul(&v)?;
        assert!(
            (attend(&q, &k, &v, 3)? - dense)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?
                < 1e-5
        );
        Ok(())
    }
}
