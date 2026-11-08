//! One neural transition per invocation. No state promotion without visual review.
use crate::{
    inference,
    weights::{self, Result},
};
use aster_core::{
    adaptive,
    interaction::{self, ReferenceFrame, ReferenceFrameManager, VisualReview},
    video::VideoSpec,
};
use candle_core::{Device, IndexOp, Tensor};
use candle_video::utils::deterministic_rng::Pcg32;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const FRAMES: usize = 49;
const CONTEXT: usize = 9;
const SEED: u64 = 43;
const REFERENCE_BYTES: usize = 3 * CONTEXT * inference::WIDTH * inference::HEIGHT * 4;

fn path(dir: &Path, stage: usize, extension: &str) -> PathBuf {
    dir.join(format!("stage-{stage:02}.{extension}"))
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let f = File::open(path)?;
    if !f.metadata()?.is_file() || f.metadata()?.len() > limit as u64 {
        return Err("artifact exceeds size limit".into());
    }
    let mut bytes = Vec::new();
    f.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("artifact grew beyond size limit".into());
    }
    Ok(bytes)
}
fn save(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

fn prompt(stage: usize) -> String {
    // Reuse the measured baseline's initial composition, without claiming metric
    // world coordinates are visually calibrated by this text description.
    if stage == 0 {
        return "A full-body adult man stands clearly visible in the foreground on a sidewalk. He has short brown hair and wears a blue jacket and black trousers. His entire body is visible from head to shoes. Beside him on the street is a yellow sedan taxi with black wheels. He faces the rear passenger door. Realistic daylight, stationary camera, side view.".into();
    }
    let action = match stage {
        0 => {
            "A full-body adult man stands clearly visible at the edge of a sidewalk. His entire body is visible from head to shoes. A yellow sedan taxi is on the road several metres away, approaching from his left. All car doors are closed."
        }
        1 => {
            "The standing man raises his right arm above his head and holds it up to hail the distant yellow taxi."
        }
        2 => "The yellow taxi drives slowly closer to the waiting man with his arm raised.",
        3 => {
            "The taxi brakes and comes to a complete stop beside the waiting man. All doors remain closed."
        }
        4 => {
            "The man lowers his arm and walks to the rear passenger door of the stationary taxi, stopping beside it."
        }
        5 => {
            "The man extends his right hand toward the rear passenger door handle. The taxi remains stationary and the door closed."
        }
        6 => {
            "The man's right hand firmly grasps the rear passenger door handle. The door is still closed."
        }
        7 => {
            "The man pulls the handle and the rear passenger door begins swinging outward, revealing the dark interior."
        }
        8 => {
            "The man pulls the rear passenger door fully open and holds it. The opening and back seat are clearly visible."
        }
        9 => {
            "The man bends his head and steps through the open rear door. His torso crosses the doorway and is partly hidden behind the door."
        }
        10 => {
            "The man completes entering the taxi and sits on the rear seat, fully inside. The rear passenger door is still open."
        }
        11 => {
            "From inside the taxi, the seated man pulls the open rear passenger door inward. The door is closing."
        }
        12 => {
            "The rear passenger door finishes closing flush with the car body. The man remains seated inside."
        }
        _ => {
            "With the man inside and all doors closed, the same yellow taxi drives away along the street. No person is left outside."
        }
    };
    format!(
        "{action} The same man has short brown hair, a blue jacket and black trousers. The same yellow four-door sedan has black wheels. Fixed side-view camera, daylight, unchanged grey city sidewalk."
    )
}

fn manifest(dir: &Path, stage: usize) -> Result<String> {
    let state = interaction::expected(stage)?;
    Ok(format!(
        "version=2\nstage={stage}\nframes={FRAMES}\ncontext={CONTEXT}\nseed={}\nprofile=ltx-0.9.8-2b-first-pass-384x256-24fps-seven-sigmas\nvideo_sha256={}\nreference_sha256={}\nprompt={}\nexpected={:?}\n",
        SEED + stage as u64,
        hash(&bounded(&path(dir, stage, "mp4"), 256 * 1024 * 1024)?),
        hash(&bounded(&path(dir, stage, "f32"), REFERENCE_BYTES)?),
        prompt(stage),
        state.current()
    ))
}

fn read_review(bytes: &[u8], digest: &str) -> Result<VisualReview> {
    let text = std::str::from_utf8(bytes)?;
    let mut values = std::collections::HashMap::new();
    for line in text.lines() {
        let (key, value) = line.split_once('=').ok_or("invalid review line")?;
        if values.insert(key, value).is_some() {
            return Err("duplicate review field".into());
        }
    }
    if values.len() != 8 || values.get("manifest_sha256") != Some(&digest) {
        return Err("review must bind current manifest and contain eight fields".into());
    }
    let decision = |name| -> Result<bool> {
        match values.get(name) {
            Some(&"pass") => Ok(true),
            Some(&"fail" | &"unknown") => Ok(false),
            _ => Err("missing/invalid visual decision".into()),
        }
    };
    let reviewer = values.get("reviewer").ok_or("missing reviewer")?;
    if reviewer.trim().is_empty() {
        return Err("empty reviewer".into());
    }
    if values.get("evidence").is_none_or(|v| v.trim().is_empty()) {
        return Err("empty visual evidence".into());
    }
    Ok(VisualReview {
        digest: digest.into(),
        action: decision("action")?,
        person: decision("person")?,
        vehicle: decision("vehicle")?,
        background: decision("background")?,
        transition: decision("transition")?,
        evidence: format!(
            "{reviewer}: {}",
            values.get("evidence").ok_or("missing evidence")?
        ),
    })
}

fn accepted(dir: &Path, stage: usize) -> Result<ReferenceFrameManager> {
    let mut manager = ReferenceFrameManager::default();
    for previous in 0..stage {
        let actual = manifest(dir, previous)?;
        if bounded(&path(dir, previous, "manifest"), 16384)? != actual.as_bytes() {
            return Err("reference manifest/artifact/settings mismatch".into());
        }
        let digest = hash(actual.as_bytes());
        let review = read_review(&bounded(&path(dir, previous, "review"), 8192)?, &digest)?;
        manager.accept(
            ReferenceFrame {
                snapshot: interaction::expected(previous)?.current().clone(),
                digest,
            },
            &review,
        )?;
    }
    Ok(manager)
}

/// Assemble only accepted stages, omitting overlapping reference frames.
pub fn export(args: &[String]) -> Result<()> {
    if args.len() != 8
        || args[0] != "--run-dir"
        || args[2] != "--ffmpeg"
        || args[4] != "--output"
        || args[6] != "--stages"
    {
        return Err(
            "control-export --run-dir DIR --ffmpeg PATH --output NEW.mp4 --stages 1..14".into(),
        );
    }
    let count: usize = args[7].parse()?;
    if !(1..=interaction::STAGES.len()).contains(&count) {
        return Err("invalid stage count".into());
    }
    let dir = Path::new(&args[1]);
    accepted(dir, count)?;
    let output = Path::new(&args[5]);
    if output.extension().and_then(|x| x.to_str()) != Some("mp4") {
        return Err("output must be MP4".into());
    }
    let mut command = std::process::Command::new(&args[3]);
    command.args(["-v", "error", "-nostdin"]);
    for stage in 0..count {
        command.arg("-i").arg(path(dir, stage, "mp4"));
    }
    let mut graph = String::new();
    let mut streams = String::new();
    for stage in 0..count {
        let skip = if stage == 0 { 0 } else { CONTEXT };
        graph.push_str(&format!(
            "[{stage}:v]trim=start_frame={skip},setpts=PTS-STARTPTS[v{stage}];"
        ));
        streams.push_str(&format!("[v{stage}]"));
    }
    graph.push_str(&format!("{streams}concat=n={count}:v=1:a=0[out]"));
    // Reserve the destination ourselves; FFmpeg never receives overwrite authority.
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let status = command
        .args([
            "-filter_complex",
            &graph,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ])
        .stdout(file)
        .status()?;
    if !status.success() {
        return Err(format!("accepted-stage assembly failed: {status}").into());
    }
    println!(
        "status={} accepted_stages={count} frames={} verification=external_manual_review reencoded=true",
        if count == interaction::STAGES.len() {
            "reviewed_scenario"
        } else {
            "partial_scenario"
        },
        FRAMES + (count - 1) * (FRAMES - CONTEXT)
    );
    Ok(())
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 8
        || args[0] != "--models"
        || args[2] != "--run-dir"
        || args[4] != "--ffmpeg"
        || args[6] != "--stage"
    {
        return Err("control --models DIR --run-dir DIR --ffmpeg PATH --stage 0..13".into());
    }
    let models = Path::new(&args[1]);
    let dir = Path::new(&args[3]);
    let ffmpeg = Path::new(&args[5]);
    let stage: usize = args[7].parse()?;
    let state = interaction::expected(stage)?;
    // Review checks occur before GPU initialization or output creation.
    let manager = accepted(dir, stage)?;
    if !ffmpeg.is_file() {
        return Err("FFmpeg missing".into());
    }
    fs::create_dir_all(dir)?;
    let _reservation = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path(dir, stage, "started"))?;
    for ext in ["mp4", "f32", "manifest"] {
        if path(dir, stage, ext).exists() {
            return Err("stage artifact already exists; use a new run directory".into());
        }
    }
    let _monitor = crate::telemetry::Monitor::start();
    let device = Device::new_cuda(0)?;
    let start = Instant::now();
    let model = models.join("ltxv-2b-0.9.8-distilled.safetensors");
    weights::verify(
        &model,
        6_340_744_492,
        "76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f",
    )?;
    weights::verify(
        &models.join("text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf"),
        3_386_856_640,
        "b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae",
    )?;
    let generation = Instant::now();
    let previous = if stage > 0 {
        let bytes = bounded(&path(dir, stage - 1, "f32"), REFERENCE_BYTES)?;
        if bytes.len() != REFERENCE_BYTES {
            return Err("wrong reference extent".into());
        }
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        if values
            .iter()
            .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
        {
            return Err("invalid reference pixels".into());
        }
        Some(Tensor::from_vec(
            values,
            (1, 3, CONTEXT, inference::HEIGHT, inference::WIDTH),
            &Device::Cpu,
        )?)
    } else {
        None
    };
    if let Some(latest) = manager.latest() {
        let changes = adaptive::detect(&latest.snapshot, state.current(), &[], false);
        if changes.global {
            return Err("global state invalidates reference".into());
        }
        println!(
            "affected={:?} canonical_manifest={} previous_manifest={} conditioning=accepted_nine_frame_video_prefix identity_binding=external_review_only",
            changes.affected,
            manager.canonical().ok_or("missing canonical")?.digest,
            latest.digest
        );
    }
    let text = prompt(stage);
    println!(
        "stage_index={stage} action={} prompt={text:?} expected={:?}",
        interaction::STAGES[stage],
        state.current()
    );
    let embedding = inference::encode_prompt(&text, models)?;
    let reference = previous
        .as_ref()
        .map(|frame| super::latent::encode_frame(&model, frame, &device))
        .transpose()?;
    let mut rng = Pcg32::new(SEED + stage as u64, 1442695040888963407);
    let latent = super::latent::denoise(
        super::latent::Request {
            model: &model,
            text: &embedding,
            reference: reference.as_ref(),
            frames: FRAMES,
        },
        &device,
        &mut rng,
    )?;
    let video = inference::decode(&model, latent, &device, &mut rng)?;
    if video.dims() != [1, 3, FRAMES, inference::HEIGHT, inference::WIDTH] {
        return Err("invalid decoded shape".into());
    }
    let spec = VideoSpec {
        width: inference::WIDTH,
        height: inference::HEIGHT,
        frames: FRAMES,
        fps_num: 24,
        fps_den: 1,
    };
    let rgb = |frame: &Tensor| -> Result<Vec<u8>> {
        Ok(inference::rgb_bytes(
            frame.permute((1, 2, 0))?.flatten_all()?.to_vec1::<f32>()?,
        )?)
    };
    if let Some(previous) = &previous {
        let a = rgb(&previous.i((0, .., CONTEXT - 1, .., ..))?)?;
        let b = rgb(&video.i((0, .., CONTEXT, .., ..))?)?;
        let mae = a
            .iter()
            .zip(b)
            .map(|(a, b)| f64::from(a.abs_diff(b)) / 255.)
            .sum::<f64>()
            / a.len() as f64;
        println!("transition_rgb_mae={mae} metric=precodec_join_not_identity");
    }
    aster_core::mp4::encode(ffmpeg, &path(dir, stage, "mp4"), spec, |writer| {
        for i in 0..FRAMES {
            let data = video
                .i((0, .., i, .., ..))
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let pixels = rgb(&data).map_err(|e| std::io::Error::other(e.to_string()))?;
            writer.write_frame(i, &pixels)?;
        }
        Ok(())
    })?;
    let frame = video
        .narrow(2, FRAMES - CONTEXT, CONTEXT)?
        .contiguous()?
        .clamp(-1_f32, 1_f32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let bytes: Vec<u8> = frame.into_iter().flat_map(f32::to_le_bytes).collect();
    save(&path(dir, stage, "f32"), &bytes)?;
    let receipt = manifest(dir, stage)?;
    save(&path(dir, stage, "manifest"), receipt.as_bytes())?;
    println!(
        "status=awaiting_visual_review manifest_sha256={} elapsed_ms={} generation_ms={} process_peak_working_set_bytes={:?} transformer_forwards=7 skipped_forwards=0",
        hash(receipt.as_bytes()),
        start.elapsed().as_millis(),
        generation.elapsed().as_millis(),
        aster_core::memory::peak_working_set_bytes()?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn review_is_bound_to_artifact_and_unknown_does_not_pass() {
        let text = "manifest_sha256=abc\naction=unknown\nperson=pass\nvehicle=pass\nbackground=pass\ntransition=pass\nreviewer=manual\nevidence=door obscured\n";
        assert!(!read_review(text.as_bytes(), "abc").unwrap().action);
        assert!(read_review(text.as_bytes(), "other").is_err());
        assert!(read_review(text.replace("door obscured", "").as_bytes(), "abc").is_err());
        assert!(read_review(format!("{text}action=pass\n").as_bytes(), "abc").is_err());
    }
}
