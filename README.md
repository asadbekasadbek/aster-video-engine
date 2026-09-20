# Aster Video Engine — Rust-only research project

Goal: build a modular, measurable, cross-platform AI video-generation engine. Development host: Windows, RTX 4070 12 GB VRAM, 64 GB RAM. Rust is the only project implementation language; native GPU drivers, system libraries and model files may be external dependencies. No Python scripts, Python inference servers or Python training pipelines.

## Start here
1. Read `AGENTS.md`, `SPEC.md`, `ROADMAP.md`, `TASKS.md`, `PROGRESS.md`, `DECISIONS.md`.
2. Inspect the host with safe read-only commands; check `rustc`, `cargo`, `git`, `nvidia-smi`, free disk and available GPU backends.
3. Research current Rust ML options and model licenses using primary documentation; log links and dates in `RESEARCH.md`.
4. Implement the first uncompleted acceptance-tested milestone, not the entire vision at once.
5. Build, test, benchmark where applicable, review, document and checkpoint progress. Repeat.

## Current implementation
`aster-core` implements a bounded deterministic CPU temporal tensor reference, Windows process peak working-set telemetry, SafeTensors header inspection and a streaming YUV4MPEG2 writer. `aster-candle` tests pinned Candle 0.11.0 CPU matrix multiplication and automatic differentiation against scalar references. `aster-cli` exposes these through the commands below. No weights or Python; CUDA and pretrained model inference are not implemented. Research and provisional decisions are in RESEARCH.md and DECISIONS.md.

## Build and run (Windows MSVC Rust 1.98.1)
```powershell
$env:CARGO_HOME = "$PWD/.cargo-cache"
# First clone only: cargo fetch --locked (dependency download needs authorization)
cargo build --workspace --release --offline
cargo run -p aster-cli --release --offline -- device
cargo run -p aster-cli --release --offline -- synthetic --width 256 --height 256 --frames 240 --seed 42
cargo run -p aster-cli --release --offline -- backend-smoke
cargo run -p aster-cli --release --offline -- inspect path/to/model.safetensors
cargo run -p aster-cli --release --offline -- demo-video new-output.y4m
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo test --workspace --offline
```

`synthetic` defaults to 64x64x4, 24 frames, seed 42 and a 64 MiB tensor budget. `--budget-mib` bounds two f32 frame buffers, not total process memory. Hard limits: 1 GiB buffers and 1 billion element updates. Output contains a deterministic checksum, elapsed nanoseconds and process-lifetime peak working set. Timing includes allocation/recurrence/checksum, excludes process startup. GPU/energy metrics are unavailable. This output is not an inference benchmark. Only Windows has been tested; other platforms return unavailable OS memory metrics.

The core callback receives one CHW frame synchronously; retaining copies is outside the pipeline budget. Atomic cancellation is supported through the Rust API; resumable jobs and a graceful CLI signal handler are not implemented. Errors have stable codes and nonzero exit status. Windows CI is configured but has not run remotely.

`inspect` reads at most a 1 MiB header, validates shapes/dtypes/contiguous offsets and file extent without loading tensor payload. Limits: 4096 tensors, 4096 bytes/name, rank 32. Parsed metadata adds bounded overhead above the header byte count. It does not verify payload hashes, approve model provenance, or execute a graph. Inspect immutable files only; concurrent modification is not a supported snapshot mechanism.

`demo-video` writes a fixed synthetic moving square: 320x180, 48 frames, 24 fps, 2 seconds. The Rust writer creates uncompressed full-range YUV420 Y4M. Existing files are never overwritten; an I/O failure may leave a partial file and returns failure. FFmpeg is only an external validation/preview tool, not required by the engine. Artifacts are ignored by Git and reproducible. This is not AI-generated media.

Dependencies were explicitly authorized for SafeTensors/serde and Candle CPU. Exact versions/checksums are in Cargo.lock, declarations in docs/dependency-licenses.md. Native C Oniguruma is a Candle tokenizer transitive dependency compiled by the existing MSVC toolchain; no Python runtime or system installation. GPU dependencies and model weights need separate approval. Candidate weight inventory: docs/sd15-candidate-manifest.md.

## Success is evidence-based
An empty skeleton is not a video generator. A complete engine must pass the acceptance tests in SPEC.md. Claims of 1080p, speed, quality, or training are prohibited without reproducible measurements and artifacts.
