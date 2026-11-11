# Architecture decision records

ADR-000: Rust-only project implementation. Status: accepted by project owner. Native CUDA/system dependencies may be wrapped by Rust. No Python code, Python scripts, or Python inference/training service.
ADR-001: First GPU target Windows/NVIDIA; abstract backends for portability. Status: provisional until M0 research.
ADR-002: Existing licensed weights first; custom training later, with small Rust-only proof of concept. Status: provisional.
ADR-003: Evidence-based optimization and bounded resource budgets. Status: accepted.

New decisions must include date, context, alternatives, consequences, and supporting sources/benchmarks.

## ADR-004 — 2026-09-20: staged CPU baseline and first inference candidate
Status: accepted for architecture; GPU/model integration provisional pending build, approved weights and measured output.

Context: no existing implementation; Windows stable Rust/MSVC present; NVIDIA driver present but CUDA toolkit not found. 12 GiB cannot be treated as all available to model weights. See RESEARCH M0.2/M0.3 primary sources.

Decision: create a dependency-free Rust core and CLI for bounded streaming synthetic f32 latents, deterministic seed, cancellation and honest telemetry. Keep platform telemetry isolated. This is a reference computation, not pretrained inference. First model backend candidate is upstream Candle, initial model SD1.5 image smoke test (per SPEC image/video allowance). Pin release plus lockfile only after feature/build review. CUDA is first accelerated target; no automatic silent GPU-to-CPU fallback.

Alternatives: Burn is stronger for later training experiments; direct CubeCL adds kernel work without providing a video model. candle-video/LTX 2B distilled is the first separate video feasibility spike but fork dependency and custom weights license need review; Wan 1.3B is an alternative with more porting work. SVD is image-conditioned and requires temporal components absent from SD image implementation.

Gates: CUDA tensor smoke test and CPU comparison before model download; approved exact weight manifest/license; local Rust image output with peak RAM/VRAM and timing before M3 completion; separate real video output for video claim. If 12 GiB fails, reduce resolution/frame count, stage/offload components and measure; if still infeasible, retain measured image MVP and explicitly block video, never replace it with synthetic animation claims.

Consequences: no ML dependency or weight download in M1/M2. CPU baseline unblocks actual builds now. Portability is an interface goal, untested on other hardware/OS. Peak RAM uses Windows process lifetime counters; synthetic GPU utilization/VRAM/energy remain unavailable, not zero-valued invented measurements.

## ADR-005 — 2026-09-20: metadata and media boundaries
Status: implemented and locally tested on Windows.

SafeTensors 0.8.0 with existing serde dependencies is pinned after explicit user approval. Parse only a bounded header using validated upstream Metadata; independently reject repeated names and check file length. Avoid unsafe mmap and whole-weight allocation for inspection. Metadata acceptance does not claim supported model operators or payload integrity. Alternative full-buffer SafeTensors parsing would load unnecessary weight data.

Use Rust std to write bounded planar YUV420 YUV4MPEG2 streams with explicit rational frame rate, order/count validation and I/O error propagation. Use already-installed FFmpeg for independent decoding and optional MP4 preview; do not bundle its GPL-enabled binary. Synthetic motion verifies the media boundary independently of blocked pretrained inference. Sources and limitations: RESEARCH M3.1/M4.1.

## ADR-006 — 2026-09-20: isolate pinned Candle feasibility test
Status: CPU build and focused tensor/autodiff test passed; full final validation recorded in PROGRESS.

User separately authorized Candle 0.11.0 CPU dependencies. Keep the probe in aster-candle; core reference does not depend on Candle. Do not enable cuda/cudnn/mkl/metal, copy DLLs or install a toolkit implicitly. Registry Candle 0.11.0 differs from browsed main: its tokenizers dependency uses onig/native C. Existing MSVC builds it without Python. A successful CPU probe establishes only tested operations, not whole-model compatibility or video quality.

## ADR-007 — 2026-09-22: world first, conservative reuse
Context: revised owner request makes persistent state central; earlier roadmap placed it after inference. Preserve validated infrastructure, promote W1–W4 independently of model approval. Use bounded immutable snapshots, fixed-point millimetres and stable opaque appearance IDs in the first Rust stage; atomic event batches prevent half-applied interactions. Alternatives: unbounded event log, immediate neural scene parsing, per-frame prompts. Consequences: bounded deterministic symbolic tests now; no physical simulation or visual identity guarantee. Camera, relationships and parser follow in W2. Partial generation requires explicit backend evidence; unknown effects force FULL. Research sources below support layout conditioning/local editing as directions, not arbitrary reusable neural compute.
## ADR-008 — 2026-09-23: real LTX trial behind explicit approval
User prioritizes actual local text-to-video before further synthetic optimization. Select LTX 0.9.8 2B distilled with CPU GGUF T5 and staged CUDA DiT/VAE as a trial; exact sources/licenses/downloads in docs/ltx-inference-approval.md. SD image probe is no longer a required prerequisite. Alternatives Wan require more Rust porting; SVD lacks direct text input. Windows fit and generation quality remain unproven. Upstream example's silent CPU fallback, automatic downloads and whole-model GPU load are unacceptable as Aster integration defaults.

Keep restricted planner explicit; it rejects unknown text, while generate validates arbitrary text but returns backend_unavailable until actual adapter exists. FULL is the only safe neural mode until conditioning and partial inference are validated. Symbolic IDs, cache policy and residency reservations do not imply neural identity or reduced GPU work. MP4 uses existing user-selected native FFmpeg with bounded frame streaming and no-overwrite output; no redistribution or system install is needed for this tested media path. Whole inference metrics must include stated process/device scope and encoder overhead separately.
## 2026-09-23 — native LTX experimental boundary
- Keep the pinned Candle fork in an isolated Cargo workspace; communicate via native worker process and CLI arguments. Preserve registry Candle baseline.
- First trial stages CPU quantized T5, CUDA BF16 DiT and CUDA VAE sequentially. Payload bounds are not full VRAM enforcement. No block streaming or selective neural generation claim.
- Preserve official explicit distilled sigmas without terminal stretching; reject non-finite VAE output before pixel conversion.

## 2026-09-23 — observed model boundary
- First real Rust-only CUDA MP4 is verified; retain the small first-pass baseline as an experimental generator, not a production-quality scene engine.
- Wire Manager::transfer_value to actual weight copies with synchronization before releasing CPU tensors. Keep activation/workspace/T5 limits explicitly outside this ledger.
- Expose taxi-v1 scene-to-text conditioning as an explicit optional mode; do not silently replace arbitrary prompts with the restricted taxi grammar. All neural denoising remains FULL; failed action adherence is not declared fixed by symbolic replay.

## 2026-09-28 — temporal reference boundary before spatial reuse
- Retain existing generate path and add explicit sequence mode. Compile core/transformer changes against pinned sources without changing shared cache files or downloading weights.
- Use last generated full-scene frame as visual reference. Encode with existing VAE and deterministic posterior mode; freeze first latent frame with token-time0 and masked scheduler updates. Recompute all transformer attention/MLP operations. This is temporal hard conditioning, not sparse spatial generation.
- ReferenceStamp validates world tick, immutable IDs/appearance handles and globals. No silhouette/influence bounds are fabricated. Camera/light/environment changes invalidate the anchor. Quality must be judged from generated video independently of stamp equality.

2026-09-28: Keep temporal reference conditioning explicitly separate from spatial compute reuse. V1 and v2 fail controlled entry and identity acceptance; do not represent lower boundary MAE as semantic quality or speed improvement. Further action control needs a concrete keyframe/pose mechanism and compatibility evidence, not unbounded seed searches.

## ADR — 2026-10-03: evidence-gated interaction experiment
Preserve generate/sequence and add explicit control command, one transition per invocation. Expected state is not observed state. Fourteen stages split the ten owner events into handle reach/grasp, opening/open, entering/seated, closing/closed. A persisted canonical reference is never replaced by a newer drifting object. Previous accepted nine-frame video context enters the same VAE/DiT backend; canonical evidence is only an evaluator reference until an actual identity-conditioning API exists.
No installed visual action model can verify the required relations. Use documented external visual review bound to hashes of video, reference pixels, expected state, prompt and settings. Missing, failed, unknown or stale reviews prevent the next inference. This is a manually supervised experiment, not autonomous semantic verification. Full-video acceptance remains incomplete if a gate fails; do not silently continue with a fictional state.
## 2026-10-03 — quality before control/efficiency
Owner priority supersedes W8 progression. Preserve scene code and baseline; add explicit25frame quality profiles. Correct normalized-space decoder noise only in the quality experiment until compared. Increase native resolution with conservative GPU admission, not unvalidated OOM trials. Do not advertise upscaling as native detail, dev13B compatibility from a config preset, or longer distilled schedules as proven quality.13BBF16 block residency is a feasibility requirement, not an efficiency milestone; do not trade quality for FP8 before establishing reference.

Five bounded trials failed visual acceptance. Retain Q3 as a provisional review candidate only; stop seed searching. Next quality work targets recommended multi-scale parity and validated13Bdev support after new-weight authorization. Unit tests and valid MP4 do not close visual acceptance.

User supplied a visual target and permitted image conditioning. Add opt-in quality --image/reference704 and reference704-fullvae; preserve historical paths. Prefer full-temporal decoder for this measured short reference candidate because it reduces visible doubling; do not extrapolate memory needs to long videos. Image-to-video and text-only quality acceptance remain distinct; no RGB source replacement or fake pan/zoom.

2026-11-08: Keep photo conditioning separate. Compare unsplit temporal VAE in text-only mode at576, then960x640 with spatial tiling retained. A sharp car without the standing person fails. Allow a separately labeled prompt-only comparison with explicit framing; retain the exact owner-prompt results.

Photo quality: keep unsplit temporal VAE and add a bounded 960x544 image profile; do not equate more pixels with retained source detail.

Exact 3s export: generate 73 causal-compatible frames, write first72 at24fps; retain full temporal decoding with 256px spatial tiles for this profile.

Use a closer real image-conditioned take and optional local learned restoration; full restoration made skin overly smooth. Preserve unenhanced generation and report native versus delivered dimensions. Larger diffusion checkpoint remains optional and was not needed for this delivered comparison.

2026-11-08: Native 960x544/73-frame image-conditioned portrait uses query-sliced self-attention across the entire key sequence to fit the RTX4070. Keep learned weights, output cadence and unsplit temporal VAE; verify small-tensor dense equivalence and judge the actual portrait frames.

2026-11-08. For the first multi-source training experiment, freeze LTX and Real-ESRGAN. Train a small channel-shared residual processor on spatial targets plus preservation of the source adjacent-frame RGB delta. Keep it experimental and separate from the generation CLI until independent evaluation justifies its use.

Keep the approved residual weights as the reference. Experimental training writes a sibling run folder and requires an explicit checkpoint choice for rendering. Motion matching did not pass the temporal quality gate.

Ship a versioned lossless pack in Git and bind training to its SHA-256. The clone entry point trains the accepted fixed-pair objective and applies the checkpoint it just saved.

2026-11-09. Default to lossless disk-backed weight streaming in the LTX generation worker. Keep resident control selectable. Preserve text and reference attention implementations, checkpoint dtype, VAE tile layout and all denoising steps. The active half of the VAE still resides in VRAM.

2026-11-09. Preserve checkpoint quantization block contents and exact token ordering when gathering T5 embeddings. Transfer same-dtype payloads directly to the target device. Convert decoded frames to F32 only for export or bounded CPU context; retain existing CPU RGB rounding and finite checks.

2026-11-09. Keep DiT prefetch opt-in. Use CPU read-ahead with a bounded channel and retain an independently budgeted subset of exact BF16 weights across denoising steps. Free both at DiT teardown. Percentages measure additional allocations against free memory at planning time; defaults leave the existing synchronous path in place.

Run the SAM2 CPU predictor in an isolated Rust crate to keep its Candle fork out of the LTX executable. Export depth, masks, captions and CoTracker descriptors as separate research outputs. Reserve DAVIS val for evaluation. Do not use auxiliary labels in RGB reconstruction until their alignment and objectives have been validated.

Use opt-in teacher-cuda for Qwen captions on this Windows host. Single-frame captions remain human-reviewable hypotheses. Keep noncommercial CoTracker outputs out of the bundled training set pending source-specific rights review.

Use a separate opt-in reference13b-704-fullvae profile with bounded streaming for the first 13B Dev trial. Keep the existing 2B profiles and text encoder. The trial uses native BF16 weights, positive conditioning and one pass at 704x384; official multi-scale rendering uses additional guidance and a learned upscaler.

Run the 0.9.8 two-pass guidance branches sequentially to bound GPU activation memory; apply STG to self-attention values at the scheduled indices. Use 256px spatial VAE tiles for the 49-frame full-temporal decode. Keep the 2B and existing single-pass 13B profiles available.
