//! CPU feasibility probe for pinned upstream Candle; no weights or GPU fallback.
use candle_core::{Device, Tensor};

#[derive(Debug)]
pub struct SmokeReport {
    pub matmul_max_abs_error: f32,
    pub gradient_max_abs_error: f32,
}

/// Compares a real Candle matmul and autodiff result with hand-calculated values.
/// Failure to meet tolerance is an error, never a successful compatibility report.
pub fn cpu_smoke() -> candle_core::Result<SmokeReport> {
    let device = Device::Cpu;
    let a = Tensor::new(&[[1_f32, 2., 3.], [4., 5., 6.]], &device)?;
    let b = Tensor::new(&[[7_f32, 8.], [9., 10.], [11., 12.]], &device)?;
    let result = a.matmul(&b)?.flatten_all()?.to_vec1::<f32>()?;
    let matmul_max_abs_error = result
        .iter()
        .zip([58., 64., 139., 154.])
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0_f32, f32::max);
    let x = candle_core::Var::new(&[1_f32, -2., 3.], &device)?;
    let gradients = x.sqr()?.sum_all()?.backward()?;
    let gradient = gradients
        .get(&x)
        .ok_or_else(|| candle_core::Error::Msg("missing gradient".into()))?
        .to_vec1::<f32>()?;
    let gradient_max_abs_error = gradient
        .iter()
        .zip([2., -4., 6.])
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0_f32, f32::max);
    if result.len() != 4
        || gradient.len() != 3
        || result.iter().chain(&gradient).any(|v| !v.is_finite())
        || matmul_max_abs_error > 1e-6
        || gradient_max_abs_error > 1e-6
    {
        candle_core::bail!("CPU tensor/autodiff parity failed");
    }
    Ok(SmokeReport {
        matmul_max_abs_error,
        gradient_max_abs_error,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn upstream_candle_cpu_matches_scalar_reference() {
        let result = super::cpu_smoke().unwrap();
        assert_eq!(result.matmul_max_abs_error, 0.0);
        assert_eq!(result.gradient_max_abs_error, 0.0);
    }
}
