# Checked taxi interaction — 2026-10-03

Status: INCOMPLETE. Real neural MP4s generated and inspected, but full scenario
failed at stopping / vehicle geometry. Door interaction and entry were NOT run
after the failed gate. Do not mark this milestone complete from the unit tests.

## Preserved baseline

The prior artifacts and temporal-reference-benchmark.md are retained unchanged:
337 frames /14.041667s,149.100s total,144.120s generation excluding integrity,
peak working set4712796160bytes, sampled device-wide VRAM5170MiB. The82.05%
join-MAE reduction belongs to the earlier v1 paired independent/reference test,
not a measured action success rate and not a speedup. No compute saving.

New runs keep model/checksums, native Rust/CUDA,49frames per invocation,
384x256/24fps, base seed43 with stage increment, seven first-pass distilled
sigmas, same CPU T5 and component staging. The new context is nine frames,
versus one in baseline. Event prompts/splitting and process boundaries differ.
A full same-workload comparison cannot be made: the new chain was halted.
Do not compare a partial video's total time with the baseline14second video
or attribute changed discontinuity solely to the longer context.

## Implemented contracts

- Opt-in World snapshot states for person/vehicle/door; atomic validation of
  CLOSED -> OPENING -> OPEN -> CLOSING -> CLOSED. State changes enter Change
  Detection even without position/pose changes. Legacy planner API retained.
- Fourteen expected microstages: standing,raise_hand,approach,stop,walk_to_door,
  reach_handle,grasp_handle,opening,open,entering,seated,closing,closed,depart.
  Explicit near-door, hand-near-handle, contact, occluded-by and inside relations.
  These are expected symbolic relations, not measured 3D geometry. Positions
  remain uncalibrated planning defaults; no physical-layout guarantee.
- ReferenceFrameManager retains canonical and latest accepted snapshots/digests,
  preserves original object IDs/appearance handles and rejects stale/failed/
  unknown evidence, skipped stages or changed provenance. Canonical appearance
  evidence is never silently replaced by a drifting latest frame.
- The Rust control worker persists bounded nine-frame F32 context, real MP4 and
  a manifest binding hashes, expected state, prompt and settings. Accepted
  context goes through VAE -> two hard latent frames -> native DiT inference.
  GPU runs observed192fixed tokens/step, exact frozen error0,7forwards4704tokens.
  All tokens still compute. Canonical evidence is for visual comparison; no
  object-specific neural identity embedding is implemented or claimed.
- External manual visual review is required before another transition. Reviews
  record action/person/vehicle/background/transition separately. SHA binding
  detects stale artifacts but does not authenticate a reviewer or make their
  judgment automatically correct. This is not an autonomous scene recognizer.
- Rust control-export validates all included reviews, removes nine overlapping
  frames from later clips and re-encodes via native FFmpeg. Fewer than14 accepted
  stages report partial_scenario. No export/inference advances through failure.

## Real runs and measurements

Files: artifacts/controlled-taxi-20261003/ (rejected initial composition),
artifacts/controlled-taxi-v2-20261003/ (reviewed chain), with corresponding logs
in artifacts/. Every generated stage contains49frames and independently decodes.
No build ran during measured inference. RAM is worker peak working set, excluding
FFmpeg. VRAM is sampled device-wide, includes other applications, and can miss
short peaks. Totals include integrity I/O; manual review time is excluded.

| Trial/stage | Total ms | Generation ms | Peak RAM bytes | Observed VRAM MiB | Join RGB MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial candidate1, rejected | 71822 | 27277 | 4261498880 | 5493 | N/A |
| v2 standing | 32624 | 27383 | 4260233216 | 5333 | N/A |
| v2 raise_hand | 34500 | 29204 | 4272140288 | 5300 | 0.02093494 |
| v2 approach | 35016 | 29722 | 4271378432 | 5350 | 0.02624400 |
| v2 stop, rejected | 35070 | 29793 | 4271898624 | 5275 | 0.03811712 |

v2 totals across4 invocations:137.210s including integrity,116.102s generation;
peak RAM4.272GB, peak sampled VRAM5350MiB. These are not full-scenario metrics.
Join MAE compares previous last RGB8 frame against new frame9 (first fresh frame),
pre-codec, divided by255. Mean over three attempted joins0.02843202; not comparable
to baseline's six different event joins as a controlled quality improvement.
Energy unavailable. Fine-grained identity score / action detector unavailable.

## Visual acceptance and failure evidence

Reviews and contact sheets sample frames0,8,16,24,32,40,48 per stage, compared with
canonical scene and previous stage. They are coarse visual judgments at384x256,
not biometric identification, calibrated object tracking or proof of every frame.

Initial candidate1 crops the man's head/face: rejected as identity reference.
Requesting stage1 returned failure before CUDA/output creation. v2 uses the exact
baseline's visible-full-body initial prompt; new decoded hashes are not bitwise
equal to the older baseline, so deterministic parity is NOT claimed.

| Owner requirement | Observed result |
| --- | --- |
| Person standing by road | PASS in v2; full-body man visible |
| Raises hand | PASS at shoulder height; extra prompt detail above head NOT met |
| Taxi approaches | Coarse image-space PASS: passenger area moves toward man; metric geometry unverified |
| Taxi stops | FAIL: stable stop not established in final samples |
| Walk / grasp handle / door opens / enters / closes / departs | NOT ATTEMPTED after failed stop gate |
| Person identity | Coarsely consistent hair/clothes/appearance through4 stages; entry identity unavailable |
| Car identity | FAIL at stop: grille/bumper/light geometry morphs, rear-like red lamps appear on front |
| Background | Broadly stable street/camera; plausible passing traffic |
| Joins | No strong sampled-frame discontinuity; low MAE does not fix wrong actions |

Accepted3/4 attempted v2 stage reviews (including initial composition), full
scenario success0/1. This is one selected initial composition, not a statistical
success rate. No evidence of successful door/entry control or full object identity.
Stage4 request after rejected stop returned failure; export including stage3
returned failure and created no output.

Accepted partial export: artifacts/controlled-taxi-v2-accepted-prefix.mp4,
H.264384x256/24fps,129frames/5.375s, ffprobe count and full FFmpeg decode pass.
It contains only standing/arm raise/approach; it is NOT the requested whole video.
Rejected stop clip remains artifacts/controlled-taxi-v2-20261003/stage-03.mp4.

## Concrete limitation and architectural next step

The inspected base LTX pipeline consumes text plus time-positioned pixel/video
conditioning. It has no hard API for ObjectId, door hinge angle, hand-handle
contact, occlusion constraints or subject-specific persistent identity. Frozen
past latents constrain the past, not future car geometry or contact physics.
These failures do not prove LTX can never depict entry; they show this interface
and these trials do not provide reliable state control. More prose alone is not
a validated fix. A second frame also must not be inserted into a late compressed
latent as if it were a causally encoded video chunk.

Proposed path: explicit scene/interaction controller -> identity-consistent
start/end keyframes with known door/pose states -> official time-positioned
multi-keyframe conditioning in Rust -> generated transition -> independent
visual state/identity check -> bounded retry or stop. Correct endpoint keyframes
must come from reviewed user/reference images, a verified identity-preserving
image editor, or an asset-backed renderer used as CONDITIONING (not substituted
for final neural video). The current engine does not yet have that endpoint
producer, pose/layout adapter or automatic visual verifier.

Official non-first single-frame conditioning uses additional tokens with their
own temporal coordinates, removed before decoding. Porting this mechanism is a
concrete compatible next experiment with existing weights, but does not solve
creation of correct same-identity endpoint images by itself. The13B pose LoRA
previously researched targets13B0.9.7, not approved2B0.9.8; do not install it as
an assumed solution. No new model weights/runtime/system dependencies downloaded.

Primary sources inspected2026-10-03: official pipeline
https://github.com/Lightricks/LTX-Video/blob/main/ltx_video/pipelines/pipeline_ltx_video.py
and distilled config
https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-2b-0.9.8-distilled.yaml .
Source conclusions versus locally verified behavior are distinguished above.
Validation: root fmt/clippy/release and36tests pass; isolated CUDA release fmt/clippy/release and9tests pass; installed native MP4 roundtrip1test pass:46total. Existing native LNK4098 warning remains. Final git diff --check passes. Source changes remain local/uncommitted.
