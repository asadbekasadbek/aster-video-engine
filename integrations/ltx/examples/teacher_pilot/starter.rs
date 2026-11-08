use super::*;
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, process::Command};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const PACK: &str = "training/packs/student-v1.mkv";

fn ffmpeg() -> Result<PathBuf> {
    env::split_paths(&env::var_os("PATH").unwrap_or_default())
        .map(|p| p.join(format!("ffmpeg{}", env::consts::EXE_SUFFIX)))
        .find(|p| p.is_file())
        .ok_or_else(|| "Install FFmpeg with libx264 and add ffmpeg to PATH.".into())
}

fn frames(ff: &Path, input: &Path, stream: usize, resize: bool) -> Result<Vec<u8>> {
    let mut cmd = Command::new(ff);
    cmd.args(["-nostdin", "-v", "error", "-i"])
        .arg(input)
        .args(["-map", &format!("0:v:{stream}")]);
    if resize {
        cmd.args(["-vf", "fps=24,scale=576:384:force_original_aspect_ratio=decrease:flags=lanczos,pad=576:384:(ow-iw)/2:(oh-ih)/2"]);
    }
    let result = cmd
        .args([
            "-frames:v",
            "25",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .output()?;
    if !result.status.success() || result.stdout.len() != 25 * F {
        return Err(format!(
            "Expected 25 decodable frames in {}: {}",
            input.display(),
            String::from_utf8_lossy(&result.stderr)
        )
        .into());
    }
    Ok(result.stdout)
}

pub(super) fn apply(checkpoint: &Path, input: &Path, output: &Path) -> Result<()> {
    let ff = ffmpeg()?;
    let net = load_net(checkpoint)?;
    render(&net, &frames(&ff, input, 0, true)?, output, &ff)
}

pub(super) fn run() -> Result<()> {
    let ff = ffmpeg()?;
    // In a partial clone Git fetches the pack blob using the configured remote.
    let pack = Path::new(PACK);
    let parent = Path::new("artifacts/student-training");
    fs::create_dir_all(parent)?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    let run = parent.join(format!("run-{stamp}"));
    fs::create_dir(&run)?;
    let path = if pack.is_file() {
        pack.to_owned()
    } else {
        let file = run.join("student-v1.mkv");
        let status = Command::new("git")
            .args(["show", &format!("HEAD:{PACK}")])
            .stdout(fs::File::create_new(&file)?)
            .status()?;
        if !status.success() {
            return Err(
                "Git could not retrieve training/packs/student-v1.mkv from this revision".into(),
            );
        }
        file
    };
    if fs::metadata(&path)?.len() != 27_181_764
        || format!("{:x}", Sha256::digest(fs::read(&path)?))
            != "b94c65421f0150d949bc4b3fefaad30d12861cf6afb6271ed0e2f666002be999"
    {
        return Err("training pack failed SHA-256 verification".into());
    }
    let clip = |stream| -> Result<Clip> {
        Ok(Clip {
            x: frames(&ff, &path, stream, false)?,
            y: frames(&ff, &path, stream + 1, false)?,
        })
    };
    let data = [clip(0)?, clip(2)?];
    let hold = clip(4)?;
    let net = train(&data, 1., false)?;
    let checkpoint = run.join("student.safetensors");
    save_net(&net, &checkpoint)?;
    let trained = load_net(&checkpoint)?;
    let b = batch(std::slice::from_ref(&hold), &mut Rng(2028), 65536, false)?;
    let (s, t) = losses(&trained.predict(&b.inputs)?, &b)?;
    let metrics = format!(
        "detail_rmse={:.5}\nadjacent_delta_rmse={:.5}\n",
        32. * s.to_scalar::<f32>()?.sqrt(),
        32. * t.to_scalar::<f32>()?.sqrt()
    );
    fs::write(run.join("metrics.txt"), &metrics)?;
    render(&trained, &hold.x, &run.join("preview.mp4"), &ff)?;
    println!(
        "{metrics}{}\n{}",
        checkpoint.display(),
        run.join("preview.mp4").display()
    );
    Ok(())
}
