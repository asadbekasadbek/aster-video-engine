use crate::{
    inference,
    weights::{self, Result},
};
use aster_core::{adaptive, continuity::ReferenceStamp, planner, video::VideoSpec};
use candle_core::{Device, IndexOp, Tensor};
use candle_video::utils::deterministic_rng::Pcg32;
use std::{path::Path, time::Instant};

const IDENTITY: &str = "A man with short brown hair wears a blue jacket and dark trousers. A yellow taxi with black wheels is beside a grey city sidewalk. Daylight, fixed wide camera, same man and taxi.";

fn action_text(action: &str) -> Result<&'static str> {
    Ok(match action {
        "initial" => {
            "A full-body adult man stands clearly visible in the foreground on a sidewalk. He has short brown hair and wears a blue jacket and black trousers. His entire body is visible from head to shoes. Beside him on the street is a yellow sedan taxi with black wheels. He faces the rear passenger door. Realistic daylight, stationary camera, side view."
        }
        "hail" => {
            "The man standing outside the taxi raises his right arm and waves to hail the approaching taxi."
        }
        "approach" => {
            "The yellow taxi moves toward the waiting man and stops beside him at the curb."
        }
        "open_door" => {
            "The man standing outside reaches for the rear passenger door handle and pulls the door open."
        }
        "enter" => {
            "The man bends down and steps through the open rear passenger door, then sits inside the taxi."
        }
        "close_door" => {
            "The man is seated inside the taxi. He pulls the rear passenger door closed."
        }
        "depart" => {
            "With the man seated inside and all doors closed, the yellow taxi accelerates away along the street."
        }
        _ => return Err("unsupported event action".into()),
    })
}

pub struct Options {
    pub frames: usize,
    pub reuse: bool,
    pub events: usize,
    pub seed: u64,
}

pub fn generate(
    prompt: &str,
    models: &Path,
    output: &Path,
    ffmpeg: &Path,
    options: Options,
) -> Result<()> {
    let Options {
        frames,
        reuse,
        events,
        seed,
    } = options;
    seed.checked_add(events as u64)
        .ok_or("event seed overflow")?;
    inference::validate_frames(frames)?;
    let mut plan = planner::plan(prompt)?;
    if events > plan.events.len() {
        return Err("events exceeds planned sequence length".into());
    }
    plan.events.truncate(events);
    if output.exists()
        || output.extension().and_then(|v| v.to_str()) != Some("mp4")
        || !ffmpeg.is_file()
    {
        return Err("need a new MP4 path and installed FFmpeg".into());
    }
    let _monitor = crate::telemetry::Monitor::start();
    let device = Device::new_cuda(0)?;
    let start = Instant::now();
    let model = models.join("ltxv-2b-0.9.8-distilled.safetensors");
    println!("stage=integrity_check");
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
    println!(
        "stage=integrity_check elapsed_ms={}",
        start.elapsed().as_millis()
    );
    let generation_start = Instant::now();
    let output_frames = frames + plan.events.len() * (frames - 1);
    let spec = VideoSpec {
        width: inference::WIDTH,
        height: inference::HEIGHT,
        frames: output_frames,
        fps_num: 24,
        fps_den: 1,
    };
    let mut last_frame: Option<Tensor> = None;
    let mut last_pixels: Option<Vec<u8>> = None;
    let mut stamp = ReferenceStamp::for_snapshot(plan.world.current());
    let mut output_index = 0;
    let mut references = 0;
    aster_core::mp4::encode(ffmpeg, output, spec, |writer| {
        let perform = || -> Result<()> {
            for phase in 0..=plan.events.len() {
                let stage = Instant::now();
                let (action, can_reference) = if phase == 0 {
                    ("initial", false)
                } else {
                    let e = &plan.events[phase - 1];
                    let before = plan.world.current().clone();
                    plan.world
                        .apply_scene(e.tick, &e.updates, Some(&e.relations), None)?;
                    let changes = adaptive::detect(&before, plan.world.current(), &[], false);
                    let valid = reuse && stamp.can_condition(&before, &changes);
                    println!(
                        "event={} world_tick={} affected={:?} temporal_reference={} spatial_mode=FULL",
                        e.action, e.tick, changes.affected, valid
                    );
                    (e.action, valid)
                };
                let text = if phase == 0 {
                    action_text(action)?.to_owned()
                } else {
                    format!("{} {IDENTITY}", action_text(action)?)
                };
                let embeddings = inference::encode_prompt(&text, models)?;
                let reference = if can_reference {
                    references += 1;
                    Some(super::latent::encode_frame(
                        &model,
                        last_frame.as_ref().ok_or("missing neural reference")?,
                        &device,
                    )?)
                } else {
                    None
                };
                let mut rng = Pcg32::new(seed + phase as u64, 1442695040888963407);
                let latents = super::latent::denoise(
                    super::latent::Request {
                        model: &model,
                        text: &embeddings,
                        reference: reference.as_ref(),
                        frames,
                    },
                    &device,
                    &mut rng,
                )?;
                let video = inference::decode(&model, latents, &device, &mut rng)?;
                if video.dims() != [1, 3, frames, inference::HEIGHT, inference::WIDTH] {
                    return Err("invalid scene video shape".into());
                }
                // The first generated reference frame overlaps the previous segment;
                // omit it equally in both reference and independent baselines.
                for frame in usize::from(phase > 0)..frames {
                    let pixels = inference::rgb_bytes(
                        video
                            .i((0, .., frame, .., ..))?
                            .permute((1, 2, 0))?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    )?;
                    if phase > 0 && frame == 1 {
                        let old = last_pixels.as_ref().ok_or("missing previous RGB frame")?;
                        let mae = old
                            .iter()
                            .zip(&pixels)
                            .map(|(a, b)| f64::from(a.abs_diff(*b)) / 255.0)
                            .sum::<f64>()
                            / pixels.len() as f64;
                        println!(
                            "event={action} boundary_rgb_mae={mae} metric_scope=continuity_not_semantics"
                        );
                    }
                    writer.write_frame(output_index, &pixels)?;
                    output_index += 1;
                    if frame == frames - 1 {
                        last_pixels = Some(pixels);
                    }
                }
                last_frame = Some(
                    video
                        .narrow(2, frames - 1, 1)?
                        .contiguous()?
                        .clamp(-1_f32, 1_f32)?,
                );
                stamp = ReferenceStamp::for_snapshot(plan.world.current());
                println!(
                    "event={action} elapsed_ms={} output_frames={output_index}",
                    stage.elapsed().as_millis()
                );
            }
            Ok(())
        };
        let mut perform = perform;
        perform().map_err(|e| std::io::Error::other(e.to_string()))
    })?;
    println!(
        "kind=event_neural_video frames={output_frames} references_encoded={references} transformer_forwards={} skipped_transformer_forwards=0 elapsed_ms={} generation_ms={} process_peak_working_set_bytes={:?}",
        7 * (plan.events.len() + 1),
        start.elapsed().as_millis(),
        generation_start.elapsed().as_millis(),
        aster_core::memory::peak_working_set_bytes()?
    );
    Ok(())
}
