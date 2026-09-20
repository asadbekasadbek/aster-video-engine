# Resumable progress log

Status: M0 research, M1, M2.1, M3.1, M3.2a CPU probe and independent M4.1 complete. Next M3.2b requires model/dependency/license approval. Pretrained inference, CUDA and actual AI video remain unverified.

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

