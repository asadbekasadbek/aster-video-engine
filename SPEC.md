# Product specification and acceptance criteria

## Vision
Local-first Rust AI video engine: prompt -> persistent scene/world state -> timeline/motion -> adaptive generation -> temporal reconstruction -> upscaling -> video file. Treat this as a research hypothesis, not proven architecture.

## Platform
Windows + RTX 4070 12 GB + 64 GB RAM is the first target. Keep filesystem, GPU, media, and model backends abstract for future Linux/macOS. The implementation must be Rust-only; external native GPU/media libraries are allowed behind explicit FFI or command adapters, with documented licenses.

## MVP acceptance
- Cargo workspace builds on Windows; `cargo test --workspace` passes.
- CLI reports device/memory capabilities and can process a reproducible tiny synthetic frame/latent pipeline on CPU.
- Rust model-loader prototype can inspect an approved, documented model format and report unsupported operations honestly.
- One supported, licensed pretrained model can produce a short actual image/video result with a Rust inference path; record exact model, version, backend and output. If no suitable Rust path exists, keep this milestone BLOCKED, do not silently add Python.
- Media output can be played and metadata (dimensions, fps, duration) verified.
- Peak VRAM and RAM plus elapsed time recorded for reproducible inputs.

## Research milestones (not guaranteed on this PC)
- Rust-only training of a tiny video/motion model on a small permissioned dataset, checkpoint and deterministic evaluation.
- Scene identity persistence and camera/motion state with testable behavior.
- Frame interpolation and selective generation compared against a full-generation baseline.
- Longer 10-second 1080p output, with transparent distinction between native generation and upscale; 24 fps output means 240 frames.
- Cross-platform CI and backend tests on actual hardware.

## Nonfunctional requirements
Bounded memory; cancellation; resumable jobs; clear errors; deterministic seeds where supported; licensed datasets/models; no credentials in logs; reproducible benchmarks; documented unsupported cases. No unverifiable 'ideal quality' claim.
