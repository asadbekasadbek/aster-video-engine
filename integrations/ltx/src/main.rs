use candle_core::{Device, Tensor};
mod inference;
mod prefetch;
mod quality;
mod reference;
mod restore;
mod stream_t5;
mod telemetry;
mod vae_half;
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
    if args.first().map(String::as_str) == Some("teacher-target") {
        return restore::prepare(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("restore-glance") {
        return restore::run(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("control") {
        return reference::controlled::run(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("control-export") {
        return reference::controlled::export(&args[1..]);
    }
    if args.len() >= 9
        && args.len() % 2 == 1
        && matches!(args[0].as_str(), "generate" | "sequence" | "quality")
        && args[1] == "--prompt"
        && args[3] == "--models"
        && args[5] == "--output"
        && args[7] == "--ffmpeg"
    {
        let mut frames = 25;
        let mut reuse = true;
        let mut events = 6;
        let mut seed = 42;
        let mut profile = "corrected";
        let mut image = None;
        let mut seen = std::collections::HashSet::new();
        for pair in args[9..].as_chunks::<2>().0 {
            if !seen.insert(&pair[0]) {
                return Err("duplicate worker option".into());
            }
            match pair[0].as_str() {
                "--frames" => frames = pair[1].parse()?,
                "--events" if args[0] == "sequence" => events = pair[1].parse()?,
                "--seed" if args[0] == "sequence" || args[0] == "quality" => {
                    seed = pair[1].parse()?
                }
                "--profile" if args[0] == "quality" => profile = &pair[1],
                "--image" if args[0] == "quality" => image = Some(std::path::Path::new(&pair[1])),
                "--reuse" if args[0] == "sequence" => {
                    reuse = match pair[1].as_str() {
                        "on" => true,
                        "off" => false,
                        _ => return Err("reuse must be on|off".into()),
                    }
                }
                _ => return Err("unknown worker option".into()),
            }
        }
        if args[0] == "quality" {
            if seen.iter().any(|flag| flag.as_str() == "--frames")
                && frames
                    != (if matches!(
                        profile,
                        "reference704-3s-fullvae" | "reference960-3s-fullvae"
                    ) {
                        73
                    } else {
                        25
                    })
            {
                return Err("frame count conflicts with quality profile".into());
            }
            return quality::generate(
                &args[2],
                args[4].as_ref(),
                args[6].as_ref(),
                args[8].as_ref(),
                profile,
                seed,
                image,
            );
        }
        if args[0] == "sequence" {
            return reference::scene::generate(
                &args[2],
                args[4].as_ref(),
                args[6].as_ref(),
                args[8].as_ref(),
                reference::scene::Options {
                    frames,
                    reuse,
                    events,
                    seed,
                },
            );
        }
        return inference::generate(
            &args[2],
            args[4].as_ref(),
            args[6].as_ref(),
            args[8].as_ref(),
            frames,
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
