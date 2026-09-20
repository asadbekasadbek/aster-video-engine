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

