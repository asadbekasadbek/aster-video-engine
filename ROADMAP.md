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

2026-09-20 dependency clarification: M4.1 raw-frame media I/O can be validated with clearly labelled synthetic data before M3 weights approval. This does not complete pretrained generation. M3 is split into header inspection, CPU library probe, actual image inference, CUDA feasibility and a separate real video experiment. M5–M9 remain future work; final SPEC acceptance is not met by these infrastructure checks.

## Revised execution order — 2026-09-22
Preserve M0–M4 results and approval gates. Promote world architecture ahead of further model integration:
1. W1 bounded persistent world and atomic events (first verifiable stage).
2. W2 camera/light/environment, relationships and validated restricted text planner for the taxi scenario.
3. W3 dependency-aware change detection and conservative mode selection with explicit backend capabilities.
4. W4 deterministic synthetic renderer/cache and adaptive-vs-FULL equality/work comparison; integrate CLI and media output. This is architectural validation, not AI video.
5. W5 approved model adapter: audit masking/conditioning/cache compatibility before partial inference; retain SD1.5 only as image probe, LTX as provisional video candidate.
6. W6 temporal quality evaluation, actual residency scheduler and native MP4 integration; reproducible neural FULL vs adaptive measurements.
7. Resume M8/M9 only after evidence supports training and higher resolution. Optimize each stage only after baseline measurement.
2026-09-23 owner priority: after W2/W3a policy and M4.2 MP4, proceed to approved W5 real video trial before additional synthetic W4 rendering. SD image fallback is not a prerequisite or completion criterion for the first text-to-video prototype. Preserve FULL mode until local/regional neural support and quality are measured.