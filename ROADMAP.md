# Roadmap

M0: Audit environment and Rust-only feasibility; research existing models/frameworks, licenses and backend support. Deliver an architecture decision with honest blockers.
M1: Cargo workspace, CLI, structured errors/logs, device abstraction, unit tests and CI.
M2: Rust tensor/latent/memory prototype; measure VRAM/RAM/latency on synthetic data.
M3: Licensed pretrained model integration with Rust-only inference, first real media output.
M4: Rust video I/O, timestamps, frame sequencing, 24 fps and output verification.
M5: Scene state, identity, timeline and motion with deterministic tests.
M6: Keyframes/interpolation, selective regeneration and quality comparisons.
M7: Memory scheduler, prefetch, quantization/tiling where supported, repeatable benchmarks.
M8: Rust-only dataset ingest, validation and tiny training experiment; checkpoints and evaluation.
M9: 10 s / 1080p target and cross-platform portability experiments; report limits and tradeoffs.

At every milestone: design -> implementation -> tests -> benchmark -> review -> documentation -> checkpoint. Advance only when acceptance evidence exists.
