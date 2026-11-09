# Aster Video Engine — Rust-only research project

Goal: build a modular, measurable, cross-platform AI video-generation engine. Development host: Windows, RTX 4070 12 GB VRAM, 64 GB RAM. Rust is the only project implementation language; native GPU drivers, system libraries and model files may be external dependencies. No Python scripts, Python inference servers or Python training pipelines.

## Обучение после git clone

Из корня репозитория запустите `cargo aster-train`. Команда обучает CPU-модуль обработки кадров на комплектном наборе и применяет сохранённые веса к проверочному видео. Нужны Rust/Cargo, системный инструментарий сборки и FFmpeg в PATH. [Команды для обучения и обработки своего ролика](docs/student-training-ru.md)

## Start here
1. Read `AGENTS.md`, `SPEC.md`, `ROADMAP.md`, `TASKS.md`, `PROGRESS.md`, `DECISIONS.md`.
2. Inspect the host with safe read-only commands; check `rustc`, `cargo`, `git`, `nvidia-smi`, free disk and available GPU backends.
3. Research current Rust ML options and model licenses using primary documentation; log links and dates in `RESEARCH.md`.
4. Implement the first uncompleted acceptance-tested milestone, not the entire vision at once.
5. Build, test, benchmark where applicable, review, document and checkpoint progress. Repeat.

## Current implementation
`aster-core` implements bounded world state, change policies, residency accounting, CPU tensor references, Windows memory telemetry, SafeTensors inspection and media export. `aster-candle` preserves the Candle 0.11.0 CPU baseline. Native Rust LTX inference now produces real MP4 on Windows/RTX4070 without Python. A 25-frame taxi clip passed independent decode and visual inspection; the longer hail/enter action sequence failed semantic acceptance. Build/run instructions and limitations are in [integrations/ltx/README.md](integrations/ltx/README.md).

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

`demo-video` writes a fixed synthetic moving square: 320x180, 48 frames, 24 fps, 2 seconds. The Rust writer creates uncompressed full-range YUV420 Y4M. Existing files are never overwritten; an I/O failure may leave a partial file and returns failure. The Y4M path does not require FFmpeg. The separate MP4 adapter below uses an explicitly selected installed FFmpeg executable. Artifacts are ignored by Git and reproducible. This is not AI-generated media.

Dependencies were explicitly authorized for SafeTensors/serde and Candle CPU. Exact versions/checksums are in Cargo.lock, declarations in docs/dependency-licenses.md. Native C Oniguruma is a Candle tokenizer transitive dependency compiled by the existing MSVC toolchain; no Python runtime or system installation. GPU dependencies and model weights need separate approval. Candidate weight inventory: docs/sd15-candidate-manifest.md.

## Success is evidence-based
An empty skeleton is not a video generator. A complete engine must pass the acceptance tests in SPEC.md. Claims of 1080p, speed, quality, or training are prohibited without reproducible measurements and artifacts.

World architecture includes bounded state/relationships, a restricted taxi CLI planner, conservative change detection and generation policy. Residency checks now gate actual RAM-to-VRAM weight transfers. Optional `--planner taxi-v1` converts replayed events into whole-clip neural text conditioning. Spatial selective compute reuse and neural identity control are not implemented. Revised milestones and evidence are in SPEC/ROADMAP/TASKS.

## Text scenario CLI (2026-09-23)
```powershell
.\target\release\aster.exe plan --prompt "Человек стоит на улице, останавливает такси, открывает дверь и садится в машину. Такси уезжает"
.\target\release\aster.exe plan --prompt-file examples/taxi.txt
```
This is a restricted Russian grammar: initial street sentence, hail, optional explicit taxi approach, open door, enter car, optional close door, taxi departure. Clauses can use commas, periods, semicolons, line breaks or the conjunction `и`. Approach/door-close are inserted if omitted and reported as planner defaults. Unknown actions/order fail. Output reports 4 objects, 6 default events and FULL decisions because no renderer influence bounds exist. It does not produce video.

`aster generate --prompt TEXT --output NEW.mp4` (or --prompt-file PATH) forwards validated text to a native LTX worker. Supply `--worker`, `--models` and `--ffmpeg` as documented in the integration README. Missing worker or failed inference returns an error; there is no substitute-media success path. The downloads and license in docs/ltx-inference-approval.md have been explicitly approved.

`aster_core::mp4::encode` accepts ordered RGB8 frames and a path to installed FFmpeg, producing H.264 fragmented MP4. The LTX worker calls it after decoding and validating model output; that complete path has passed real model execution and independent decoding. Standard tests do not require FFmpeg; the native integration test can be run explicitly:
```powershell
$env:ASTER_TEST_FFMPEG = 'C:\OSPanel\modules\php\php_8.2\ffmpeg.exe'
$env:ASTER_TEST_FFPROBE = 'C:\OSPanel\modules\php\php_8.2\ffprobe.exe'
cargo test -p aster-core --test mp4 --locked --offline -- --ignored --nocapture
```
The residency manager gates weight transfers during component loading; it does not yet manage activation/workspace allocation, T5, or complete tensor lifetimes. Adaptive spatial modes are decision policy only, with neural partial generation disabled.

Experimental `aster sequence` connects world/change decisions to real previous-frame VAE conditioning. Join discontinuity improves in the measured ablation, but door opening, entry and visual identity remain unverified/failed; no transformer compute is saved. See [temporal reference measurements](docs/temporal-reference-benchmark.md).

`aster control` adds evidence-gated taxi microtransitions with explicit door/person/vehicle states and nine-frame neural video-prefix conditioning. It retains canonical appearance evidence; visual reviews are external/manual, not an automatic action model. `control-export` assembles only accepted stages. Current trial stops at failed taxi stopping/geometry; full scenario remains incomplete. See [checked interaction report](docs/checked-interaction-benchmark.md).
# Current priority: one-second visual quality

Use `aster quality` for bounded real 25-frame LTX trials. See
[quality comparison](docs/quality-baseline.md) for the reproducible command,
five generated clips, settings and measurements. The photorealism acceptance
criterion is **not yet met**. World/action and efficiency work is deferred.

Windows x64 CPU student training: see [Russian quick start](docs/student-training-ru.md).

Для эксперимента с предзагрузкой на Windows скопируйте [.env.example](.env.example) в .env и запустите генерацию из корня проекта. Настройки RAM и VRAM описаны в [инструкции LTX](integrations/ltx/README.md).
