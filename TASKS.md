# Task queue

- [x] M0.1 Record hardware, Windows build toolchain, Rust version, available GPU backend, disk space (no installs without approval).
- [x] M0.2 Research Burn/Candle/CubeCL and Rust-compatible video model architectures; verify training vs inference support and CUDA/Windows compatibility. Source evidence and untested cases: RESEARCH 2026-09-20.
- [x] M0.3 Research candidate pretrained video models, weights formats, licenses, VRAM requirements and Rust porting effort. Four candidates in RESEARCH; no model approved/downloaded.
- [x] M0.4 Write ADR for first inference backend and model; include fallback if 12 GB is insufficient. ADR-004.
- [x] M1.1 Create Cargo workspace and documented crates after M0 decisions. Offline Windows build passes.
- [x] M1.2 Add minimal CLI, structured errors, unit tests, fmt/clippy/test workflow. Local validation passes; remote CI not executed.
- [x] M2.1 Implement bounded-memory synthetic tensor pipeline and instrument peak memory. Release evidence in PROGRESS.
- [x] M3.1 Bounded safetensors inspector: dependencies approved; metadata and offsets validated without tensor allocation; malformed/truncated/oversized fixtures rejected; no graph-execution claims. Core and CLI tests pass.
- [x] M3.2a Pinned Candle 0.11.0 CPU feasibility probe after explicit dependency approval: actual matmul and autodiff agree with independent scalar values to <=1e-6. Measured errors both zero on Windows.
- [ ] M3.2b Local pretrained image inference. BLOCKED on next approval: candle-nn/candle-transformers and image/tokenizer dependencies, exact weight/tokenizer download and license decision (docs/sd15-candidate-manifest.md). Implement local-only adapter; generate and inspect real image; measure RAM/time. CPU path can proceed without CUDA installation, with CUDA metrics explicitly unavailable.
- [ ] M3.2c CUDA smoke/telemetry. BLOCKED: CUDA Toolkit not located; installation and CUDA dependencies need approval. After build, compare GPU tensors to CPU and record process/device VRAM scope correctly.
- [x] M4.1 Independent media boundary: Rust YUV4MPEG2 streaming writer, strict frame order/count/shape, 24 fps synthetic test clip, independent ffprobe/decoder validation. Can precede M3 because it consumes raw frames; does not satisfy pretrained inference.
- [ ] M3.3 Actual video model feasibility: audit/pin candle-video fork, license/weights approval, short LTX 2B trial and measured memory/quality. Depends on M3.2 and separate GPU feasibility evidence; do not claim it from synthetic media.

For each task add acceptance criteria and evidence before checking it off. Append tasks for later milestones only after dependencies are understood.

## Acceptance criteria (2026-09-20)
- M0.1: independently record OS/CPU/RAM/GPU/disk/toolchain, distinguish driver from toolkit, inspect MSVC; explicit unavailable backend status is acceptable for an audit. Evidence: PROGRESS audit checkpoint.
- M0.2: primary-source matrix for all three libraries covering training/inference, exact observed versions/features, build dependencies, Windows limits, video architectures, 12 GiB constraints; distinguish documentation from local tests.
- M0.3: compare at least three video models and an image fallback; record weight format, license source, memory evidence and missing Rust components; no weight download required.
- M0.4: dated ADR chooses first backend/model, separates provisional GPU choice from tested CPU baseline, defines reduced fallback and validation gates.
- M1.1: documented core/CLI Cargo crates; Windows build succeeds offline without third-party dependencies.
- M1.2: strict help/device/synthetic CLI, stable error codes, unit/integration tests and CI workflow; fmt/clippy/workspace tests pass.
- M2.1: deterministic streaming temporal f32 prototype, checked sizes and memory budget, cancellation, process peak working-set measurement on Windows; reproducible release workload/checksum, reject invalid/overflow/budget inputs; report GPU metrics as unavailable for CPU workload.


## Priority override — world architecture (2026-09-22)
Existing completed tasks remain valid. Work in this order before blocked M3 downloads:
- [x] W1 Bounded world state and atomic events. Acceptance: unique stable IDs/appearance; deterministic object order; strictly increasing ticks; no partial mutation on unknown/duplicate participants; unchanged objects preserved; history eviction and hard limits tested; fmt/clippy/workspace tests pass.
- [x] W2 Add camera/light/environment and bounded relationships; restricted taxi text planner with explicit unsupported-input errors. Test event order, same actor/vehicle identity, door/contact/containment transitions and deterministic replay. Do not present a template grammar as general NLP.
- [ ] W3 Change reports with old/new bounds, interaction closure and conservative camera/light invalidation; mode selection with backend capability fallback; test all four modes and unknown effects.
- [ ] W4 Rust synthetic adaptive/full renderer comparison, CLI scene trace and media path. Same workload/seed must produce identical frames; measure actual reused/computed pixels, time and RAM. No neural speedup claims.
- [ ] W5 Audit and implement approved neural conditioning/masking/cache path; depends on separate dependency/weights/license approval. Actual AI output and quality evidence required.
- [ ] W6 Integrated MP4, resource residency scheduler and paired neural metrics; do not mark complete from synthetic tests.
## 2026-09-22 implementation checkpoint and inference priority
- [x] W2 Restricted Russian taxi planner + CLI --prompt/--prompt-file, camera/light/environment revisions, bounded relationships, explicit implicit approach/door close and unknown-action/order errors. World and deterministic replay tests pass.
- [x] W3a Change set + old/new influence union + relationship closure; four-mode policy guarded by exact cache conditioning and backend capabilities. Missing visual influence information forces FULL. Synthetic projection/rendering and measured reuse remain W3/W4 unfinished.
- [x] W6a Residency ledger with SSD/RAM/VRAM budgets, adjacent transfer callbacks, simultaneous source/destination/workspace accounting and rollback tests. Actual model offload/VRAM enforcement remains unfinished.
- [x] M4.2 Native MP4 adapter: stream ordered RGB8 frames into explicitly selected existing FFmpeg; no overwrite, child cleanup, H.264 decode/metadata verification. Encoder test is synthetic and NOT AI output.
- [ ] W5.1 APPROVAL REQUIRED: pinned LTX Rust/CUDA dependency build, CUDA 12.6.3 Toolkit-only installation, LTX license acceptance and 9.73 GB weights/tokenizer manifest. Exact request: docs/ltx-inference-approval.md. No download/install has occurred.
- [ ] W5.2 After approved numerical CUDA probe, stage model components and connect actual LTX inference to generate CLI and MP4. Measure RAM/VRAM/time, evaluate own English text and taxi sequence visually. Current generate returns backend_unavailable without creating output.

The user's latest priority is real text-to-video after approval; W4 synthetic renderer is no longer a prerequisite for W5. Keep W3/W4 research open without postponing approved inference to add further test animations.
## 2026-09-23 authorization and execution update
- [x] W5.1a User approved CUDA, pinned model downloads and LTX license. Toolkit installed without driver update; Rust CUDA release build and numerical GPU probe passed.
- [ ] W5.1b Complete active weight downloads and verify pinned SHA256 values.
- [ ] W5.2 Execute native Rust model, reject non-finite decoded values, export and inspect real MP4; record actual memory/time. CLI worker forwarding implemented; model test still pending.

- [x] W5.1b Approved downloads complete; native worker verified both model SHA256 values before entering T5. CUDA-feature strict release clippy also passed.

- [x] W5.2a Native Rust arbitrary text -> real LTX inference -> finite frames -> MP4, independently decoded and visually inspected. First experiment25frames; exact memory/time in PROGRESS.
- [ ] W5.2b Longer hail/enter taxi semantic test (121frames) and inspection, then connect world/residency policies to measured model computation.

- [x] W6b Connect residency budget validation to real RAM->VRAM weight transfers; typed backend failure rollback test, live LTX replay and exact decoded-frame parity. Scope/metrics in PROGRESS; full model residency and workspace budgeting remain open.
- [ ] W5.2b Taxi semantic acceptance remains FAILED despite valid AI MP4. Wire restricted scene state/event conditioning next; do not claim text requests enforce identity/actions.

- [x] W5.3a Restricted planner->World State replay->change decisions->whole-clip neural text conditioning wired and executed. Semantic action test remains FAILED; does not implement per-event neural scheduling or identity binding.
- [ ] W5.3b Exact T5 conditioning cache: bounded storage, model/tokenizer/version/prompt invalidation, corrupt-cache rejection, paired real inference with identical decoded frames and measured time. This only skips unchanged text encoding, not video denoising.

## 2026-09-28 owner priority: event-conditioned neural continuity
- [x] W7.1 Exact expanded Russian taxi scenario -> ordered validated events; immutable appearance descriptors and IDs; reject unknown grammar. Tests cover requested sequence.
- [x] W7.2 Native per-token timestep adapter with parity against existing scalar-time path and malformed-shape tests. Real hard reference tokens must remain unchanged at every denoising step.
- [ ] W7.3 Event loop: world/change decisions govern reference invalidation and actual VAE-encoded prior-frame conditioning. Compare conditioned vs independent events with same prompts/seeds/frames; report measured boundary errors, operation counts, RAM/VRAM/time and visual event/identity acceptance. Fixed latents alone do not count as saved transformer work.

- [x] W7.1 Exact control scenario and immutable snapshot reference provenance tests pass. Six compound events retained (arm raise+hail, approach+stop, rear-door opening, entry, close, departure).
- [x] W7.2 Token-time adapter scalar parity, nonuniform-time sensitivity, shape rejection and fixed-prefix invariance tests pass. Actual GPU hard-token invariance is verified in W7.3a.

- [x] W7.3a Real GPU temporal hard reference verified:6VAE anchors,96fixed tokens/step,error0; paired independent/reference337frame clips decoded and measured. Mean joinMAE reduced82.05percent in v1, transformer work unchanged, segment latency +2.58percent. See docs/temporal-reference-benchmark.md.
- [ ] W7.3b Semantic/identity acceptance FAILED in v1 and v2: v2 raises arms but does not open the door or enter; taxi body drifts. Initial seed43 contains person+taxi. Equal-frame baseline comparison recorded separately. Next: investigate explicit door/body keyframe or pose conditioning compatible with the approved 2B model; do not repeat unbounded prompt/seed trials.

## 2026-10-03 — checked interaction transitions
- [ ] W8.1 Add explicit typed door/person/vehicle states and spatial relations; atomic invalid-state rejection and change detection tests. Preserve old scene API.
- [ ] W8.2 Bounded Reference Frame Manager with immutable canonical identity reference, recent accepted reference, hash-bound visual review and fail-closed progression. Do not infer visual success from symbolic state.
- [ ] W8.3 Real Rust/CUDA stepwise taxi experiment, same49frame/384x256/24fps/seven-step settings as baseline, inspect every transition before next; preserve failures and measure actual outcomes. Full scenario stays incomplete if a gate fails.

- [x] W8.1 Typed states/door chain/spatial relations and atomic state-aware change detection tests passed.
- [x] W8.2 Canonical/latest reference metadata and hash-bound fail-closed visual review wired to real persisted frames. External manual evaluator only; no automatic recognition or proven neural identity binding. Failed initial reference blocked stage1 in a real CLI check.

- [ ] W8.3 INCOMPLETE after real GPU trials: initial/head crop rejected; v2 standing,raised arm,coarse approach accepted; stop rejected for continued motion/car geometry. No door/entry test after failure. Native reviewed-prefix export129frames verified. Full same-settings baseline comparison is unavailable because the chain stopped. See docs/checked-interaction-benchmark.md.
- [ ] W8.4 Next: port official non-first keyframe token conditioning and establish a legitimate identity-consistent endpoint producer plus visual evaluator. Do not insert still-image latents as a fake late video frame or assume13B control weights match2B. Existing initial-frame/short-video paths remain preserved.

## 2026-10-03 — user priority override: one-second photorealism
World/door/taxi action control, long clips and efficiency optimization are deferred. Preserve existing paths. Q1 acceptance: actual25frame~1s MP4 on supplied simple prompt, audit model/config/scheduler/VAE/T5/precision against primary sources, compare bounded configurations, save all settings and memory/time, visually assess realism. Quality remains incomplete until visual criteria pass. New large weight downloads require a concrete manifest and confirmation. Stop after presenting a satisfactory quality baseline; optimize only after user confirms.

- [x] Q1a Audit official2B/13B inference, add bounded Rust quality CLI, correct VAE noise order in isolated profiles, test and run five real25frame configurations. Independent decode/metadata pass. Report: docs/quality-baseline.md.
- [ ] Q1b Photorealistic human+taxi acceptance FAILED in all five trials. Q3 is provisional only. No efficiency/control progression. Await requested approval for new13Bdev/upscaler files; then implement exact multi-scale path and bounded13B support before new quality comparisons. Do not silently equate a13B preset with compatible inference.

- [ ] Q2 User-photo reference quality: real image-to-video at704x384/25frames. Acceptance: neural motion with same face/jacket/taxi, retained fine texture and sharp foreground, stable geometry; record input hash, crop, settings and peaks. Image conditioning is a separate experiment, not proof of text-only quality. Compare seeds42/43 and inspect native frames; do not count repeated source frames as neural motion.

- [x] Q2a Real image conditioning and three measured neuralMP4 trials completed;47standardtests pass. Same-seed full-temporal VAE decode visibly reduces middle-frame doubling. Selected candidate i2v-704-fullvae42.mp4; docs/photo-quality-baseline.md.
- [ ] Q2b Equal-photo-detail acceptance remains open: fine texture loss and composition drift persist. Present candidate for review; do not start efficiency stage or count this as text-only baseline success.

## 2026-11-08: text-only quality first
- Compare the supplied prompt at seed42 with Q3 settings, changing only temporal VAE splitting, then change native resolution alone to 960x640. Save MP4, settings and telemetry and inspect human detail, car, lighting, motion and morphing. No image, no optimization, no control events. Visual acceptance remains open.

- [x] Generated and visually compared Q5/Q6/Q7. Q7 (explicit medium-shot prompt, native960x640, seed42) selected for owner review. Quality acceptance and efficiency stage remain pending.

- Photo follow-up: compare full-temporal decode at 960x544 against 704x384 on the same source, prompt and seed; inspect focus on face, hands, cloth and car in moving frames.

- [x] Two photo960 outputs fully decoded and visually compared; photo960-tap selected for review. Fine-texture acceptance remains open.

- Photo request: create one continuous real image-conditioned scene, export exactly 72 frames at 24fps, fully decode it and inspect the first, middle and last frames before delivery.

- [x] Three-second photo glance: 72 frames decoded and gaze transition visually inspected; MP4 delivered.

- Three-second moving-face quality: compare learned restoration at equal display size and regenerate a closer portrait from the owner image if the eye artifact survives restoration. Select a real MP4 by visual inspection, with weights and full settings recorded.

- [x] New portrait take, equal-scale SR comparison, final verified MP4 selected for owner review. No efficiency progression.

- Produce a continuous 3s photo-conditioned 960x544 native portrait, keep full temporal decoding and full image attention, inspect gaze/hands and fine detail against prior 704x384 render before choosing output.

- [x] Native 960x544 three-second portrait generated, visually reviewed and fully decoded; delivered without post sharpening.

- [x] Train a Rust residual student with one spatial and one temporal supervision signal. Compare with spatial-only training on a separate clip, save weights and decode the output.
- [ ] Test on diverse held-out motions and scenes with independent evaluation before considering generator-level distillation.

- [x] Compare local patch tracking against frozen student and smoke-test a one-click local Windows x64 training archive. Preserve baseline when temporal RMSE regresses.

- [x] Reproduce the approved student and render through one Cargo alias after cloning the repo. Verify a separate local partial clone with the pack absent from its checkout.

- [x] Stream T5 and DiT blocks and load only the required VAE half. Verify identical MP4 for text and photo, record resource cost against resident mode.

- [x] Reduce redundant CPU allocations in stream-mode T5 lookup and same-dtype weight uploads. Compare individual changes at fixed seed and verify RGB export against the original text MP4.

- [x] Implement opt-in block read-ahead and separate available RAM/VRAM percentage budgets. Verify exact output, timing and memory on real text and photo generation and document PowerShell/.env usage.
