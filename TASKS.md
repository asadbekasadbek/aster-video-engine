# Task queue

- [x] M0.1 Record hardware, Windows build toolchain, Rust version, available GPU backend, disk space (no installs without approval).
- [ ] M0.2 Research Burn/Candle/CubeCL and Rust-compatible video model architectures; verify training vs inference support and CUDA/Windows compatibility.
- [ ] M0.3 Research candidate pretrained video models, weights formats, licenses, VRAM requirements and Rust porting effort.
- [ ] M0.4 Write ADR for first inference backend and model; include fallback if 12 GB is insufficient.
- [ ] M1.1 Create Cargo workspace and documented crates after M0 decisions.
- [ ] M1.2 Add minimal CLI, structured errors, unit tests, fmt/clippy/test workflow.
- [ ] M2.1 Implement bounded-memory synthetic tensor pipeline and instrument peak memory.

For each task add acceptance criteria and evidence before checking it off. Append tasks for later milestones only after dependencies are understood.

