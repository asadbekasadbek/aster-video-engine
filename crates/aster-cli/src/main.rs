use aster_core::{Config, Error};
use std::io::{self, Write};
use std::sync::atomic::AtomicBool;

const HELP: &str = "Aster CPU reference prototype (no model inference)\n\
Usage: aster help | device | synthetic [--width N] [--height N] [--channels N] \
[--frames N] [--seed N] [--budget-mib N]\n\
Defaults: 64x64x4, 24 frames, seed 42, 64 MiB tensor budget.\n\
Media test: aster demo-video NEW_PATH.y4m (320x180, 48 frames, 24 fps).\n\
Model metadata: aster inspect PATH.safetensors (no inference).\n\
Backend check: aster backend-smoke (Candle CPU matmul and autodiff).\n\
Output: key=value records; timing includes allocation and checksum.\n";

fn config(args: &[String]) -> Result<Config, Error> {
    let mut c = Config::default();
    let mut seen = std::collections::HashSet::new();
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        if !seen.insert(flag) {
            return Err(Error::Invalid("duplicate option"));
        }
        let value = args.next().ok_or(Error::Invalid("option needs a value"))?;
        if flag == "--seed" {
            c.seed = value
                .parse()
                .map_err(|_| Error::Invalid("seed must be a u64"))?;
            continue;
        }
        let n: usize = value
            .parse()
            .map_err(|_| Error::Invalid("option value must be an unsigned integer"))?;
        match flag.as_str() {
            "--width" => c.width = n,
            "--height" => c.height = n,
            "--channels" => c.channels = n,
            "--frames" => c.frames = n,
            "--budget-mib" => c.budget_bytes = n.checked_mul(1024 * 1024).ok_or(Error::Overflow)?,
            _ => return Err(Error::Invalid("unknown option")),
        }
    }
    c.validate()?;
    Ok(c)
}

fn execute(args: &[String], out: &mut impl Write) -> Result<(), Box<dyn std::error::Error>> {
    match args.first().map(String::as_str) {
        None | Some("help" | "--help" | "-h") if args.len() <= 1 => write!(out, "{HELP}")?,
        Some("device") if args.len() == 1 => {
            writeln!(
                out,
                "backend=cpu\nos={}\narch={}\nlogical_cpus={}\ncuda=not_implemented\ngpu_memory=unavailable",
                std::env::consts::OS,
                std::env::consts::ARCH,
                std::thread::available_parallelism()?
            )?;
            match aster_core::memory::peak_working_set_bytes()? {
                Some(bytes) => writeln!(out, "process_peak_working_set_bytes={bytes}")?,
                None => writeln!(out, "process_peak_working_set_bytes=unavailable")?,
            }
        }
        Some("backend-smoke") if args.len() == 1 => {
            let start = std::time::Instant::now();
            let report = aster_candle::cpu_smoke()?;
            writeln!(
                out,
                "backend=candle_cpu\nversion=0.11.0\nmatmul_max_abs_error={}\ngradient_max_abs_error={}\nelapsed_ns={}\ncuda=not_tested\nmodel_inference=not_tested",
                report.matmul_max_abs_error,
                report.gradient_max_abs_error,
                start.elapsed().as_nanos()
            )?;
        }
        Some("demo-video") if args.len() == 2 => {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&args[1])?;
            aster_core::video::write_demo(io::BufWriter::new(file))?;
            writeln!(
                out,
                "kind=synthetic_video\nformat=yuv4mpeg2\nwidth=320\nheight=180\nframes=48\nfps=24\nduration_seconds=2"
            )?;
        }
        Some("inspect") if args.len() == 2 => {
            let mut file = std::fs::File::open(&args[1])?;
            if !file.metadata()?.is_file() {
                return Err(Box::new(Error::Invalid(
                    "model path must be a regular file",
                )));
            }
            let inspected = aster_core::model::inspect(&mut file)?;
            let names = inspected.metadata.offset_keys();
            writeln!(
                out,
                "format=safetensors\nheader_bytes={}\ndata_bytes={}\ntensors={}\ninference=unsupported\noperators=unknown_format_contains_no_graph\npayload_integrity=not_checked",
                inspected.header_bytes,
                inspected.data_bytes,
                names.len()
            )?;
            for name in names {
                if let Some(info) = inspected.metadata.info(&name) {
                    writeln!(
                        out,
                        "tensor={name:?} dtype={:?} shape={:?} offsets={:?}",
                        info.dtype, info.shape, info.data_offsets
                    )?;
                }
            }
        }
        Some("synthetic") => {
            let c = config(&args[1..])?;
            let report = aster_core::run(c, &AtomicBool::new(false), |_, _| Ok(()))?;
            let peak = aster_core::memory::peak_working_set_bytes()?;
            writeln!(
                out,
                "kind=synthetic_temporal_latents\nbackend=cpu\nwidth={}\nheight={}\nchannels={}\nframes={}\nseed={}\ntensor_bytes={}\nchecksum={:016x}\nelapsed_ns={}\ngpu_peak_vram=unavailable\ngpu_utilization=unavailable\nenergy=unavailable",
                c.width,
                c.height,
                c.channels,
                report.frames,
                c.seed,
                report.tensor_bytes,
                report.checksum,
                report.elapsed.as_nanos()
            )?;
            match peak {
                Some(bytes) => writeln!(out, "process_peak_working_set_bytes={bytes}")?,
                None => writeln!(out, "process_peak_working_set_bytes=unavailable")?,
            }
        }
        _ => {
            return Err(Box::new(Error::Invalid(
                "unknown command or extra arguments; use help",
            )));
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match execute(&args, &mut io::stdout().lock()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            let code = error.downcast_ref::<Error>().map_or_else(
                || {
                    if error.is::<io::Error>() {
                        "io_error"
                    } else {
                        "backend_error"
                    }
                },
                Error::code,
            );
            let _ = writeln!(
                io::stderr().lock(),
                "level=error code={code} message={error:?}"
            );
            std::process::ExitCode::from(1)
        }
    }
}
