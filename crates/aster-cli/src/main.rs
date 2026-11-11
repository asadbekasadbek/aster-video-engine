use aster_core::{Config, Error};
use std::io::{self, Write};
use std::sync::atomic::AtomicBool;

const HELP: &str = "Aster Video Engine experimental CLI\n\
Usage: aster help | device | synthetic [--width N] [--height N] [--channels N] \
[--frames N] [--seed N] [--budget-mib N]\n\
Defaults: 64x64x4, 24 frames, seed 42, 64 MiB tensor budget.\n\
Media test: aster demo-video NEW_PATH.y4m (320x180, 48 frames, 24 fps).\n\
Model metadata: aster inspect PATH.safetensors (no inference).\n\
Backend check: aster backend-smoke (Candle CPU matmul and autodiff).\n\
Scene planning: aster plan --prompt TEXT | --prompt-file PATH (restricted taxi grammar).\n\
Generation: aster generate --prompt TEXT|--prompt-file PATH --output NEW.mp4 \
[--worker PATH] [--models DIR] [--ffmpeg PATH] [--frames N] [--planner taxi-v1].\n\
Event experiment: aster sequence --prompt TEXT|--prompt-file PATH --output NEW.mp4 \
[--worker PATH] [--models DIR] [--ffmpeg PATH] [--frames N] [--reuse on|off] \
[--events 0..6] [--seed N] (restricted taxi grammar; frames per segment).\n\
Checked taxi: aster control --worker PATH --models DIR --run-dir DIR --ffmpeg PATH --stage 0..13.\n\
Accepted export: aster control-export --worker PATH --run-dir DIR --ffmpeg PATH --output NEW.mp4 --stages 1..14.\n\
Quality: aster quality --prompt TEXT|--prompt-file PATH --output NEW.mp4 [--worker PATH] [--models DIR] [--ffmpeg PATH] [--profile legacy|corrected|native576|native576-t5-256|reference13b-multiscale-2s|reference13b-704-fullvae|reference704|reference704-fullvae|reference704-3s-fullvae|reference960-fullvae|reference960-3s-fullvae|text576-fullvae|text960-fullvae] [--seed N] [--image PATH].\n\
Output: key=value records; timing includes allocation and checksum.\n";

fn prompt_argument(args: &[String]) -> Result<String, Box<dyn std::error::Error>> {
    use std::io::Read;
    if args.len() != 2 {
        return Err(Box::new(Error::Invalid(
            "provide exactly --prompt TEXT or --prompt-file PATH",
        )));
    }
    let prompt = match args[0].as_str() {
        "--prompt" => args[1].clone(),
        "--prompt-file" => {
            let file = std::fs::File::open(&args[1])?;
            if !file.metadata()?.is_file() {
                return Err(Box::new(Error::Invalid(
                    "prompt path must be a regular file",
                )));
            }
            let mut text = String::new();
            file.take(aster_core::planner::MAX_PROMPT_BYTES as u64 + 1)
                .read_to_string(&mut text)?;
            text
        }
        _ => return Err(Box::new(Error::Invalid("unknown prompt option"))),
    };
    aster_core::planner::validate_prompt(&prompt)?;
    Ok(prompt)
}

fn generate(args: &[String], mode: &str) -> Result<(), Box<dyn std::error::Error>> {
    let sequence = mode == "sequence";
    let quality = mode == "quality";
    if args.len() < 4
        || args[2] != "--output"
        || !args[3].ends_with(".mp4")
        || !args.len().is_multiple_of(2)
    {
        return Err(Box::new(Error::Invalid(
            "use generate --prompt TEXT|--prompt-file PATH --output NEW.mp4 [--worker PATH] [--models DIR] [--ffmpeg PATH] [--frames N] [--planner taxi-v1]",
        )));
    }
    let mut prompt = prompt_argument(&args[..2])?;
    let mut worker = std::env::current_exe()?.with_file_name(if cfg!(windows) {
        "aster-ltx.exe"
    } else {
        "aster-ltx"
    });
    let mut models = std::path::PathBuf::from("models/ltx-0.9.8-2b");
    let mut ffmpeg = None;
    let mut frames = None;
    let mut planner = None;
    let mut reuse = "on";
    let mut events = None;
    let mut seed = None;
    let mut profile = None;
    let mut image = None;
    let mut seen = std::collections::HashSet::new();
    for option in args[4..].as_chunks::<2>().0 {
        if !seen.insert(&option[0]) {
            return Err(Box::new(Error::Invalid("duplicate generate option")));
        }
        match option[0].as_str() {
            "--worker" => worker = (&option[1]).into(),
            "--models" => models = (&option[1]).into(),
            "--ffmpeg" => ffmpeg = Some(std::path::PathBuf::from(&option[1])),
            "--frames" => frames = Some(&option[1]),
            "--planner" => planner = Some(option[1].as_str()),
            "--events" if sequence => events = Some(&option[1]),
            "--seed" if sequence || quality => seed = Some(&option[1]),
            "--profile" if quality => profile = Some(&option[1]),
            "--image" if quality => image = Some(&option[1]),
            "--reuse" if sequence => {
                reuse = option[1].as_str();
                if reuse != "on" && reuse != "off" {
                    return Err(Box::new(Error::Invalid("reuse must be on|off")));
                }
            }
            _ => return Err(Box::new(Error::Invalid("unknown generate option"))),
        }
    }
    if sequence {
        if planner.is_some() {
            return Err(Box::new(Error::Invalid(
                "sequence uses its own event planner",
            )));
        }
        aster_core::planner::plan(&prompt)?;
    }
    if quality && planner.is_some() {
        return Err(Box::new(Error::Invalid(
            "quality trial uses raw prompt, not event planner",
        )));
    }
    if let Some(planner) = planner {
        if planner != "taxi-v1" {
            return Err(Box::new(Error::Invalid("supported planner: taxi-v1")));
        }
        let scene = aster_core::conditioning::taxi(&prompt)?;
        println!("planner=taxi-v1 original_prompt={prompt:?} visual_identity_guaranteed=false");
        for (tick, action, changes, decision) in scene.events {
            println!(
                "scene_tick={tick} action={action} affected={:?} mode={:?} reason={}",
                changes.affected, decision.mode, decision.reason
            );
        }
        prompt = scene.text;
        println!(
            "neural_conditioning={prompt:?} conditioning_scope=whole_clip_text partial_neural_generation=false"
        );
    }
    if !worker.is_file() {
        return Err(Box::new(Error::BackendUnavailable(
            "LTX worker is not built; build integrations/ltx with cuda and supply --worker PATH",
        )));
    }
    let ffmpeg = ffmpeg
        .or_else(|| {
            std::env::var_os("PATH").and_then(|path| {
                std::env::split_paths(&path)
                    .map(|p| {
                        p.join(if cfg!(windows) {
                            "ffmpeg.exe"
                        } else {
                            "ffmpeg"
                        })
                    })
                    .find(|p| p.is_file())
            })
        })
        .ok_or(Error::BackendUnavailable(
            "FFmpeg not found; supply --ffmpeg PATH",
        ))?;
    let mut command = std::process::Command::new(worker);
    command
        .args([mode, "--prompt", &prompt, "--models"])
        .arg(models)
        .args(["--output", &args[3], "--ffmpeg"])
        .arg(ffmpeg);
    if let Some(frames) = frames {
        command.args(["--frames", frames]);
    }
    if sequence {
        command.args(["--reuse", reuse]);
        if let Some(events) = events {
            command.args(["--events", events]);
        }
        if let Some(seed) = seed {
            command.args(["--seed", seed]);
        }
    }
    if quality {
        if let Some(seed) = seed {
            command.args(["--seed", seed]);
        }
        if let Some(profile) = profile {
            command.args(["--profile", profile]);
        }
        if let Some(image) = image {
            command.args(["--image", image]);
        }
    }
    // CUDA DLL discovery is explicit for this tested Windows toolkit; no install
    // or CPU fallback occurs here. CUDA_PATH can select another installed toolkit.
    #[cfg(windows)]
    {
        let cuda = std::env::var_os("CUDA_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6".into());
        let bin = cuda.join("bin");
        if bin.is_dir() {
            let mut paths = vec![bin];
            paths.extend(std::env::split_paths(
                &std::env::var_os("PATH").unwrap_or_default(),
            ));
            command.env("PATH", std::env::join_paths(paths)?);
        }
    }
    let status = command.status()?;
    if !status.success() {
        return Err(format!("LTX inference failed: {status}").into());
    }
    Ok(())
}

fn checked_worker(worker: &str) -> Result<std::process::Command, Box<dyn std::error::Error>> {
    let command = std::process::Command::new(worker);
    #[cfg(windows)]
    let command = {
        let mut command = command;
        let cuda = std::env::var_os("CUDA_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6".into());
        let mut paths = vec![cuda.join("bin")];
        paths.extend(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        ));
        command.env("PATH", std::env::join_paths(paths)?);
        command
    };
    Ok(command)
}

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
        Some("control-export") => {
            if args.len() != 11 || args[1] != "--worker" {
                return Err(Box::new(Error::Invalid(
                    "control-export --worker PATH --run-dir DIR --ffmpeg PATH --output NEW.mp4 --stages 1..14",
                )));
            }
            let status = checked_worker(&args[2])?
                .arg("control-export")
                .args(&args[3..])
                .status()?;
            if !status.success() {
                return Err(format!("reviewed export failed: {status}").into());
            }
        }
        Some("control") => {
            if args.len() != 11
                || args[1] != "--worker"
                || args[3] != "--models"
                || args[5] != "--run-dir"
                || args[7] != "--ffmpeg"
                || args[9] != "--stage"
            {
                return Err(Box::new(Error::Invalid(
                    "control --worker PATH --models DIR --run-dir DIR --ffmpeg PATH --stage 0..13",
                )));
            }
            aster_core::interaction::expected(args[10].parse()?)?;
            let mut command = checked_worker(&args[2])?;
            command.arg("control").args(&args[3..]);
            let status = command.status()?;
            if !status.success() {
                return Err(format!("checked transition failed: {status}").into());
            }
        }
        Some("plan") => {
            let prompt = prompt_argument(&args[1..])?;
            let mut plan = aster_core::planner::plan(&prompt)?;
            writeln!(
                out,
                "kind=symbolic_plan\nplanner=restricted_taxi_v1\nneural_inference=false\nimplicit_actions=approach_and_close_door_if_omitted\nprompt={:?}\nobjects={}\nevents={}",
                plan.prompt,
                plan.world.current().objects().len(),
                plan.events.len()
            )?;
            for event in plan.events {
                let before = plan.world.current().clone();
                plan.world
                    .apply_scene(event.tick, &event.updates, Some(&event.relations), None)?;
                let changes =
                    aster_core::adaptive::detect(&before, plan.world.current(), &[], false);
                let key = aster_core::adaptive::CacheKey {
                    model_revision: "none".into(),
                    conditioning: plan.prompt.clone(),
                    seed: 42,
                    frame: event.tick,
                };
                let decision = aster_core::adaptive::select(
                    &changes,
                    aster_core::adaptive::Capabilities::default(),
                    None,
                    &key,
                );
                writeln!(
                    out,
                    "generation_mode={:?} reason={} affected={:?}",
                    decision.mode, decision.reason, changes.affected
                )?;
                writeln!(
                    out,
                    "tick={} action={} participants={:?} relations={:?}",
                    event.tick,
                    event.action,
                    event.updates.iter().map(|u| u.id.0).collect::<Vec<_>>(),
                    event.relations
                )?;
            }
        }
        Some("generate") => {
            generate(&args[1..], "generate")?;
        }
        Some("sequence") => {
            generate(&args[1..], "sequence")?;
        }
        Some("quality") => {
            generate(&args[1..], "quality")?;
        }
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
