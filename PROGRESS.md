# Resumable progress log

Status: real native Rust LTX text-to-MP4 validated on Windows/RTX4070. Approved CUDA/weights installed and verified. Live weight residency gating and restricted world-to-text conditioning connected. Complex taxi action adherence FAILED; spatial reuse and identity control incomplete. Latest checkpoints contain exact measurements and next task.

## Historical checkpoint supplied at session start (claims re-audited below)
- Completed: CPU, RAM, GPU, VRAM, OS build, Rust versions and disk space checked.
- Current task: M0.1.
- System details:
  - OS: Windows 10.0.26200.9457 (Windows 11)
  - GPU: NVIDIA GeForce RTX 4070 (12282 MiB VRAM, approximately 12 GiB)
  - CPU: 13th Gen Intel(R) Core(TM) i7-13700K
  - Integrated GPU: Intel(R) UHD Graphics 770
  - RAM: 64 GB total
  - Storage:
    * C: drive - 675.38 GB used, 255.14 GB free
    * D: drive - 0.21 GB used, 1862.81 GB free (project location)
  - Rust toolchain:
    * rustc 1.98.1 (48a229cea 2026-09-01)
    * cargo 1.98.1 (797e8a9bc 2026-08-05)
- Known blockers: Rust-only inference support for candidate video models is unverified; training at scale on 12 GB VRAM is not assumed feasible.
- Next action: Research Burn/Candle/CubeCL and Rust-compatible video model architectures; verify training vs inference support and CUDA/Windows compatibility.

Append dated checkpoints: task, files changed, commands/results, measured metrics, blockers, next exact step. Never overwrite the only useful checkpoint.

## 2026-09-20 — independent audit and research checkpoint
- Initial Git: clean, commit cdc55a5; only documentation/configuration, no Cargo workspace or inference artifacts.
- rustc -Vv: 1.98.1 (48a229cea, 2026-09-01), x86_64-pc-windows-msvc; cargo 1.98.1; rustup stable default. rustfmt/clippy installed.
- nvidia-smi: RTX 4070, 12282 MiB total, 802 MiB device-wide used at observation; driver 610.62, CUDA UMD 13.3. This is not a toolkit or kernel execution test.
- CIM: i7-13700K; physical memory 68470525952 bytes; Windows build 26200, UBR 9457. Disk free bytes: C 324656082944, D 2000176943104 (point-in-time).
- vswhere found VS 2022 Community with VC.Tools.x86.x64. cl/nvcc absent from PATH; CUDA_PATH empty; standard CUDA toolkit directory absent. Nonstandard installations not excluded. Real MSVC linking will be tested by M1.
- PowerShell 7 profile/modules fail to load. Use exec login=false and .NET filesystem calls; Windows PowerShell 5.1 CIM works. No system changes made.
- CUDA build/runtime smoke test blocked pending toolkit discovery or approved installation. Weights not downloaded. Framework compatibility is source evidence only.
- Next: finish framework/model ledger and ADR, then implement dependency-free CPU baseline and validate Windows linking.

## 2026-09-20 — M0.2 through M2.1 validated
- Added primary-source library/model matrix and ADR-004. Research completeness does not mean local GPU compatibility.
- Added aster-core (bounded temporal f32 pipeline, cancellation, errors, Windows memory FFI), aster-cli (strict commands/options, diagnostics), Cargo workspace/lockfile and Windows CI workflow.
- `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --offline -- -D warnings`, `cargo test --workspace --offline`: pass, 6 core tests + 1 CLI integration test. `cargo build --workspace --release --offline`: pass. MSVC linking now actually verified. Remote CI not run.
- Release `aster synthetic`: seed 42, 64x64x4, 24 frames, tensor payload 131072 bytes, checksum 83a41ddf454b9de4, 407800 ns, process lifetime peak working set 4112384 bytes.
- Release `aster synthetic --width 256 --height 256 --frames 240`, seed 42: two runs 60217400 / 60955200 ns, checksum d04861de7ac076c5 both, tensor payload 2097152 bytes, process peak 6070272 / 6074368 bytes. Includes allocation, recurrence and checksum; excludes process startup. No video/model quality interpretation. No optimization claim or statistically robust benchmark.
- Review: checked shape/work overflow, payload budget versus OS peak distinction, sink error/cancellation propagation, Windows FFI ABI; no unsafe tensor operations, no dependencies/downloads. Prototype limits 1 GiB payload and 1 billion updates. Cancellation available in library, not a graceful CLI Ctrl-C handler; resumable jobs remain future work.
- Found existing ffmpeg/ffprobe in C:/OSPanel/modules/php/php_8.2. Next: independent M4.1 streaming media writer and decoder validation while M3.1 permission is pending.

## 2026-09-20 — validated resumable checkpoint after M3.1, M3.2a and M4.1
- User explicitly approved SafeTensors 0.8.0 plus its dependencies, then Candle-core 0.11.0 plus CPU dependencies. Cache confined to D:/Projects/aster-video-engine/.cargo-cache; models/toolkit not downloaded or installed. Cargo.lock includes exact package checksums; 142 resolved registry package declarations inventoried in docs/dependency-licenses.md (includes target-specific dependencies).
- Implemented core/model.rs bounded header inspection, core/video.rs streaming Y4M writer, CLI inspect/demo-video, new aster-candle CPU probe and backend-smoke. README/RESEARCH/DECISIONS/TASKS and CI updated. Actual registry Candle manifest differs from web main, noted in research.
- Final `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --offline -- -D warnings`, `cargo test --workspace --offline`: pass; 16 tests (13 core, 2 CLI, 1 Candle). `cargo build --workspace --release --offline`: pass. Set CARGO_HOME as below for all Cargo commands. No remote CI/push. MSVC emits informational import-library creation messages as rustc linker_messages warnings during test/release linking; no clippy warnings or test failures remain.
- `target/release/aster.exe backend-smoke`: Candle CPU 0.11.0, matrix max absolute error 0, gradient max absolute error 0, elapsed 223900 ns. This tiny probe is neither trained model nor inference throughput evidence. Native Oniguruma transitive dependency builds with existing MSVC; no Python runtime.
- Final linked CLI synthetic 256x256x4/240/seed42 rerun after adding Candle: 63545600 / 63424100 ns; checksum d04861de7ac076c5 unchanged; tensor bytes 2097152; process peak working set 6160384 / 6164480 bytes. Different linked binary from initial baseline; no optimization comparison claimed.
- `artifacts/synthetic-320x180-24fps.y4m`: Rust output, 48 frames, 320x180, 24/1 fps, 2.000000 seconds confirmed with ffprobe -count_frames. `ffmpeg -v error -i ... -f null -` decodes all frames successfully. External FFmpeg produced artifacts/synthetic-preview.mp4 and midpoint PNG; MP4 metadata matches; midpoint visually inspected. These ignored artifacts are synthetic test media, not pretrained output. Regenerate with `aster demo-video NEW_PATH.y4m`; create_new prevents overwriting existing files.
- Fixed test fixture capacity for injected video write failure; strict clippy requested simpler integer comparison in header count guard. Reran focused/full checks successfully. Review checked poison-on-I/O-error, order/count/size guards, no-overwrite CLI, immutable-file inspection assumption, escaped model names, missing graph disclaimer, finite-value checking and no model/download side effects.
- M3.2b candidate manifest saved with SD repo revision/LFS SHA-256 and tokenizer Git blob identity, total 2134829551 bytes; no files downloaded and no license accepted. Needs user authorization for model crates and download/license before starting. CPU-first image can avoid toolkit installation; CUDA remains separate blocked gate. No actual model quality, peak VRAM, training, long video, resume or portability claim.
- Next exact setup: `$env:CARGO_HOME = 'D:\Projects\aster-video-engine\.cargo-cache'`; `cargo test --workspace --locked --offline`. Then, only after approval, add pinned candle-nn/candle-transformers 0.11.0 and required tokenizer/image crates, inspect actual registry SD APIs, implement local-path-only adapter, download/verify approved manifest and run batch=1 256x256 seed=42 image with documented scheduler/steps and process-memory timing. Without approval leave M3.2b incomplete. Subsequent M3.3 video experiment must remain distinct from image/synthetic milestones.


## 2026-09-22 — W1 validated, architecture priority revised
- Read owner attachment and mandatory project context; initial tree clean at 3b89fab. Preserved existing implementation and approval gates. Updated SPEC/ROADMAP/TASKS for persistent world -> events -> change detection -> adaptive generation, with neural vs synthetic evidence separated. ADR-007 and primary-source research recorded.
- Implemented aster_core::world: immutable unique identities and appearance handles, fixed-point positions, symbolic poses, atomic multi-object updates, strictly increasing ticks, deterministic ID order, <=256 objects and <=64 historical snapshots. History count is bounded, not a measured process RAM budget; allocator failure is not recovered by this initial API. No physical simulation, rendering, general NLP or neural identity claim.
- Validation: cargo fmt --all -- --check PASS; cargo clippy --workspace --all-targets --locked --offline -- -D warnings PASS; cargo test --workspace --locked --offline PASS (19 tests: 16 core including 3 new world tests, 2 CLI, 1 Candle). Existing informational MSVC import-library linker warnings remain. git diff --check PASS; reviewed new file and documentation. No new dependencies, weights, installs, remote actions or benchmark claims.
- W1 acceptance met. Overall engine remains incomplete. Next unblocked task W2: extend world with bounded camera/light/relationships then implement restricted taxi grammar and deterministic event replay; unsupported text must fail explicitly. W3/W4 follow before blocked model integrations.
- Resume command: $env:CARGO_HOME = 'D:\Projects\aster-video-engine\.cargo-cache'; cargo test -p aster-core world --locked --offline . Read world.rs and W2 criteria before edits. Working changes are not committed.
## 2026-09-22 — W2, W3a, W6a and M4.2 implementation stages
- W2: Rust planner.rs accepts documented restricted Russian taxi clauses, preserves prompt, creates 4 stable objects, plans 6 events by default (hail/approach/open/enter/close/depart). Optional explicit approach and door close; invalid/unknown/reordered/incomplete prose rejected. world.rs now snapshots camera/light/environment and bounded relationship sets atomically. CLI plan supports direct text or <=16 KiB UTF-8 regular file. No generic NLP claim.
- W3a: adaptive.rs computes changed identities plus closure through old/new relations and unions renderer-supplied old/new influence rectangles. All four selection modes tested. Production CLI supplies no proven influence bounds, so it reports FULL with a reason. There is no actual neural cache or saved computation yet.
- W6a: residency.rs checks three independent tier budgets and transient overlap/workspace, invokes synchronous caller transfer only after validation, commits accounting only on success. Unit tests include rollback, overflow and invalid transitions. These are reservations, not measured device memory or actual model offload.
- M4.2: mp4.rs streams RGB8 through a user-selected native FFmpeg into create_new output. Validates shape/order/count; closes stdin and checks encoder exit; kills/reaps child on failure. Fragmented H.264 MP4 supports stdout. Native integration test passed using existing C:/OSPanel/modules/php/php_8.2/{ffmpeg,ffprobe}.exe: 32x32, 24 frames, 24/1 fps, 1.000000 seconds, complete independent decode; no-overwrite tested. Test colors are only an encoder fixture, not model output.
- Tests as of stage checkpoint: 29 standard tests passed (24 core, 3 CLI, 1 Candle, 1 cross-module taxi integration); optional native MP4 test additionally passed when explicitly enabled. cargo fmt --all -- --check, strict workspace clippy --locked --offline and release build passed. Existing MSVC import-library informational linker warnings persist. More final review below if applicable.
- Added generate argument validation and explicit backend_unavailable failure; arbitrary valid text does not create a file or claim generation. Taxi integration currently tests symbolic pipeline only; actual taxi video/quality and neural RAM/VRAM/time remain unavailable.
- Current selection: LTX-Video 0.9.8 2B distilled, CPU GGUF T5, staged CUDA denoiser then VAE. Pinned source/weight identities, exact sizes, dependency declarations, Toolkit-only installation and license request in docs/ltx-inference-approval.md. Upstream whole-map GPU loading and silent CPU fallback must not be reused. No model/Cargo dependency or toolkit downloads performed.
- Next: obtain approval for manifest; then isolated pinned Rust/CUDA probe before large weights. Do not claim W5 or the full prototype complete. Existing uncommitted work retained; no push/commit performed.
## 2026-09-23 — final review before permission gate
- Release CLI manually ran plan --prompt-file examples/taxi.txt: 4 objects, 6 ordered events through tick 145, unchanged environment, all modes FULL because visual influence is unknown. Prompt preserved and output explicitly neural_inference=false.
- Final review checked bounded prompt read, stable errors, atomic relationship/global updates, removed-edge dependency closure, old/new influence coverage, cache frame/version invalidation, overflow/overlap accounting and no shell invocation or output overwrite. Native MP4 subprocess has cleanup on error; external encoder RAM is not included in in-process counters. Allocation exhaustion in world snapshots and physical/semantic validity of arbitrary low-level relation graphs remain outside this initial symbolic implementation.
- No inference peak RAM/VRAM/time can be reported: model is absent and CUDA execution not tested. Existing GPU memory observation is not a generation benchmark. Completed tests are architectural/encoder tests, not taxi visual quality.
- Approval is required by the user's explicit request and AGENTS.md for dependency installs, large downloads, license acceptance and system changes. Proposed scope: pinned Rust CUDA sources/transitives; Toolkit 12.6.3 components only (no driver); manifest weights/tokenizer + custom LTX license. After approval, start with a build/numerical CUDA probe before downloading 9.73 GB weights. Stop if pin resolution or SDK support fails; do not silently substitute Python or CPU inference.
- Resume: read docs/ltx-inference-approval.md and latest TASKS, then set CARGO_HOME=D:/Projects/aster-video-engine/.cargo-cache. Current validated workspace command: cargo test --workspace --locked --offline. No commit or remote actions; all changes remain reviewable in working tree.

## 2026-09-23 — authorized LTX integration in progress
- User approved Rust/Cargo sources/dependencies, then explicitly confirmed CUDA installer, 9.73 GB model/tokenizer downloads and LTX license through async answer. Do not re-ask for this scope. Install only Toolkit runtime/compiler components, no driver.
- integrations/ltx is an isolated Cargo workspace with pinned video ebe43c4 and Candle fork d8e9292 in Cargo.lock (425 resolved packages including local core). CPU cargo check/clippy/test passed; 3 tests. Existing main workspace preserved. Added real T5/DiT/VAE staged inference code and per-tensor bounded loader; no weights or inference run yet. New MSVC LNK4098 runtime warning needs review.
- CUDA installer target/downloads/cuda_12.6.3_561.17_windows.exe downloaded: 3227678920 bytes, MD5 57B4C58801CA5D11BA8F966F149065E3 matches NVIDIA. Signature verification ongoing (PowerShell7 module failure; use powershell.exe -NoProfile with no outer variable interpolation). Installer NOT executed yet.
- Source inspection found CUDA backend requires cuRAND as part of CUDA runtime alongside cuBLAS/NVRTC. Use selected subpackages only; no default full install.
- Next install verified package, build --features cuda then numerical GPU probe; only then download approved weights. Main CLI generate is still backend_unavailable until forwarding wired. Session CPU build 52075 may have completed; download session78834 completed. No model outputs exist.

## 2026-09-23 — CUDA validated, approved weights downloading
- CUDA 12.6.3 installer verified: NVIDIA Authenticode Valid, MD5 matches official checksum. Selected compiler/runtime/cuBLAS/NVRTC/cuRAND components installed successfully (exit 0); no driver selected, driver remains 610.62. nvcc V12.6.85.
- CUDA release adapter built using MSVC 14.42.34433 via NVCC_CCBIN. RTX 4070 numerical probe: matmul max absolute error 0; BF16 pointwise, softmax and convolution pass. Full neural inference not yet tested.
- Approved downloads started: LTX session 36486, T5 session 9537. Tokenizer downloaded and Git blob identity verified. Main CLI now forwards arbitrary text to native worker; strict workspace clippy passes.
- W5.2 acceptance: verified weight hashes, real model execution, finite decoded frames, independently decodable MP4, measured process RAM/device-wide sampled VRAM/time; visual quality inspected separately. Current first experiment 384x256, 25 frames, 7 distilled steps, seed42. No neural output yet.

- Current validation: root fmt/clippy/test/release pass (29 standard tests, native MP4 test skipped this rerun). Isolated adapter fmt/clippy/test pass (5 tests); CUDA release rebuilt and numerical GPU probe passes after strict component selection fix. Partial model header is 145216 bytes and fits existing inspector bound. Downloads resumed after curl18; current sessions 17718 (LTX), 9751 (T5).

## 2026-09-23 — first real model execution started
- Both approved model downloads complete; curl exit0. Independent T5 SHA256 matches manifest. CUDA-feature release clippy passed.
- Main CLI running in session32447: target/release/aster.exe generate --prompt 'A yellow taxi drives slowly along a city street in daylight.' --output artifacts/ltx-first.mp4 --worker target/ltx/release/aster-ltx.exe --models models/ltx-0.9.8-2b --ffmpeg C:/OSPanel/modules/php/php_8.2/ffmpeg.exe . Worker passed both strict weight size/SHA256 checks and entered stage=text_encoding device=cpu. Output and inference measurements not yet available. Resume by polling session32447.

## 2026-09-23 — first real AI MP4 validated
- artifacts/ltx-first.mp4 generated through main Rust CLI and native CUDA LTX worker from arbitrary English taxi prompt. 25 frames, 384x256, 24fps, seed42, 7 steps. Full ffmpeg decode exit0; ffprobe H264/25frames/1.041667s. Contact sheet first/middle/last visually shows yellow taxi and urban street consistent with prompt; no full action/identity quality claim.
- Measured elapsed 22151ms (includes hash and loading after CUDA creation); T5 9815ms, DiT total4606ms, VAE2345ms. Process lifetime peak RAM4260130816 bytes; sampled device-wide VRAM5406MiB, baseline1052MiB, 85samples/no errors, meanGPU30.5059percent. External FFmpeg RAM excluded, exact VRAM spikes and energy unavailable.
- Added bounded --frames 8n+1 in9..121, default25. Main29tests and isolated CUDA release6tests, strict clippy and release builds pass. Started longer taxi action prompt from examples/taxi-neural.txt, target artifacts/ltx-taxi-121.mp4. First real video milestone met; W5.2 full taxi quality remains pending and selective generation is not implemented.

## 2026-09-23 — taxi quality failure and first live residency integration
- 121frame taxi prompt in examples/taxi-neural.txt produced artifacts/ltx-taxi-121.mp4: full decode pass, H264384x256/24fps/121frames/5.041667s. Semantic acceptance FAIL: contact sheet shows taxi with occupant, not the requested hail/open/enter sequence. No visual identity or event-control claim.
- 121frame metrics:31404ms total; T5 9811ms, DiT9459ms, VAE6760ms; RAM4261302272 bytes; devicewide sampledVRAM6473MiB, baseline1035,119samples/noerrors,meanGPU52.7899percent.
- Manager::transfer_value now preserves typed backend errors/results. LTX loader registers real CPU tensor payload and gates synchronous RAM->CUDA copy through residency manager, releasing source only after successful copy/synchronize. Scope is component weight loading only, not T5/activations/full lifetime tracking or SSD eviction.
- Paired25frame replay artifacts/ltx-residency.mp4 decoded frame MD5s exactly match first video. Weight-residency peaks RAM50331648/VRAM3846770944 bytes for DiT; RAM226492416/VRAM2493828324 for VAE. Measured total21653ms, processRAM4261154816, sampledGPU5400MiB. Build/tests overlapped part of this run: timings are not a controlled optimization comparison; no speedup claim. Root30tests and CUDA adapter6tests pass. Release rebuild initially hit running exe Windows lock; succeeded after worker exited.

## 2026-09-23 — symbolic conditioning connected, semantic limitation confirmed
- Added core conditioning.rs and CLI --planner taxi-v1. Restricted Russian planner replays World State and computes Changes/Full decisions before emitting whole-clip English action conditioning into actual T5. Event trace includes IDs/ticks; no spatial mask, frame schedule or identity binding is passed to model. Arbitrary raw prompt mode remains available.
- Actual --planner taxi-v1 --prompt-file examples/taxi.txt --frames121 -> artifacts/ltx-world-taxi.mp4 succeeded. Full independent decode passes; contact sheet again FAILS hail/enter semantic sequence. Total30926ms,T5 9842,DiT8915,VAE6791;RAM4260700160bytes;sampledVRAM6384MiB,baseline1037,118samples,noerrors. This demonstrates wiring, not successful scene control.
- Root31tests (26core+3CLI+1Candle+1taxi) passed; fmt/clippy/release pass. No active commands. Next bounded task: cache identical T5 conditioning with exact model/tokenizer/version identity, then compare same prompt/seed uncached/cached actual decoded frames and latency. Spatial selective generation remains unsupported until verified conditioning/masking path exists.

## Final resumable checkpoint for this work cycle
- No active downloads/inference jobs. Main release rebuilt successfully; root fmt/clippy/test31 pass and diff-check pass. CUDA adapter last validated6tests/clippy/release; source core conditioning addition is only used by parentCLI, not worker path. LNK4098 remains documented, no flags suppressed.
- Review: output no-overwrite, bounded prompt/frame/header/tensor sizes, explicit CUDA/noCPU fallback, pinned checksums, finite pixel validation, source-release synchronization, backend error preservation, conservative Full policy. Native generated artifacts decode; first/residency frame MD5 equality verified. No commits/pushes.
- Next command: set CARGO_HOME=D:/Projects/aster-video-engine/.cargo-cache; inspect integrations/ltx/src/inference.rs and TASKS W5.3b before implementing bounded exact T5 cache. Do not mark full architecture complete: per-event/keyframe neural control, object identity, spatial computation reuse, activation/full lifetime residency, complete dependency-license inventory and independent review remain outstanding. User authorization for existing installed CUDA and downloaded weights persists.
