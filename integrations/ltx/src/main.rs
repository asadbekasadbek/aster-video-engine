use candle_core::{Device, Tensor};
mod inference;
mod telemetry;
mod weights;

fn probe(device: &Device) -> candle_core::Result<f32> {
    let a = Tensor::from_vec(vec![1_f32, 2., 3., 4., 5., 6.], (2, 3), device)?;
    let b = Tensor::from_vec(vec![7_f32, 8., 9., 10., 11., 12.], (3, 2), device)?;
    let values = a.matmul(&b)?.flatten_all()?.to_vec1::<f32>()?;
    let mut error = 0_f32;
    for (actual, expected) in values.into_iter().zip([58_f32, 64., 139., 154.]) {
        if !actual.is_finite() {
            candle_core::bail!("nonfinite probe output");
        }
        error = error.max((actual - expected).abs());
    }
    if error > 1e-5 {
        candle_core::bail!("matmul error {error}");
    }
    // Exercise CUDA PTX, BF16 conversion, softmax and convolution used by LTX.
    let data = Tensor::new(&[1_f32, 2., 3., 4.], device)?;
    let transformed = data
        .to_dtype(candle_core::DType::BF16)?
        .sqr()?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1::<f32>()?;
    if transformed != [1., 4., 9., 16.] {
        candle_core::bail!("BF16 pointwise parity failed");
    }
    let softmax =
        candle_nn::ops::softmax(&Tensor::zeros((1, 4), candle_core::DType::F32, device)?, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
    if softmax.iter().any(|v| (*v - 0.25).abs() > 1e-6) {
        candle_core::bail!("softmax parity failed");
    }
    let input = data.reshape((1, 1, 2, 2))?;
    let kernel = Tensor::ones((1, 1, 1, 1), candle_core::DType::F32, device)?;
    let convolved = input
        .conv2d(&kernel, 0, 1, 1, 1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if convolved != [1., 2., 3., 4.] {
        candle_core::bail!("conv2d parity failed");
    }
    Ok(error)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if (args.len() == 9 || (args.len() == 11 && args[9] == "--frames"))
        && args[0] == "generate"
        && args[1] == "--prompt"
        && args[3] == "--models"
        && args[5] == "--output"
        && args[7] == "--ffmpeg"
    {
        return inference::generate(
            &args[2],
            args[4].as_ref(),
            args[6].as_ref(),
            args[8].as_ref(),
            if args.len() == 11 {
                args[10].parse()?
            } else {
                25
            },
        );
    }
    let device = match args.as_slice() {
        [command, backend] if command == "probe" && backend == "cpu" => Device::Cpu,
        [command, backend] if command == "probe" && backend == "cuda" => Device::new_cuda(0)?,
        _ => return Err("use aster-ltx probe cpu|cuda".into()),
    };
    let error = probe(&device)?;
    println!(
        "kind=numerical_backend_probe\ndevice={device:?}\nmatmul_max_abs_error={error}\nbf16_pointwise=pass\nsoftmax=pass\nconv2d=pass\nmodel_inference=not_tested"
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("level=error message={error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cpu_matmul_matches_scalar_values() {
        assert_eq!(super::probe(&candle_core::Device::Cpu).unwrap(), 0.0);
    }
}
