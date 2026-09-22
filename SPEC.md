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

## World State Architecture revision — 2026-09-22
The primary product is a persistent world engine, not independent frame prompting. Existing CPU/model/media milestones remain infrastructure evidence. An image smoke test alone does not satisfy the product acceptance criteria.

Module contracts (planned unless explicitly implemented):
- World State Manager: stable object/appearance IDs, positions, poses, relationships, camera, lighting, environment and bounded history. Atomic timestamped events update participants together. Initial Rust world module supports fixed-point positions, symbolic poses and bounded snapshots; relationships/camera/lighting remain next work.
- Scene Planner: prompt -> validated world plus ordered events with explicit participants. First use a documented restricted grammar; reject unsupported text. General language understanding requires a separately validated local model.
- Change Detection: two snapshots -> changed IDs, old/new projected bounds and dependency closure. Occlusion, contact, shadows/reflections and camera/light changes must invalidate affected regions; unknown influence forces FULL.
- Adaptive Generation: change report + backend capabilities + cache provenance -> REUSE/LOCAL/REGIONAL/FULL. REUSE requires identical conditioning and valid cached output; LOCAL handles isolated changes, REGIONAL handles interacting objects, FULL handles global/unknown effects. Unsupported partial operations fall back explicitly to FULL. A mask alone is not evidence of fewer model operations.
- Temporal Consistency: carry appearance references, correspondence and occlusion state; measure identity drift and temporal artifacts, not just symbolic ID equality.
- Memory Manager: account for SSD/RAM/VRAM residency, activation workspace, transfers and eviction. Implement staging/offload/quantization only for verified backend operators and measure actual peaks.
- Video Output: ordered frames/timestamps -> playable MP4 through an explicit native adapter; existing Rust Y4M is an intermediate, existing external MP4 preview is not integrated engine MP4 output.

First verifiable stage W1: <=256 immutable unique object identities, integer millimetre positions, symbolic poses, strictly increasing ticks, atomic multi-object updates, <=64 retained snapshots. Reject invalid inputs without mutation. No neural or physical-consistency claim.

Subsequent acceptance: restricted taxi scenario plans hailing, vehicle approach, door interaction, entry and departure while retaining IDs; dependency-aware change detection covers revealed background and coupled interactions. A deterministic synthetic renderer compares adaptive vs FULL byte-for-byte before a neural backend. Then run identical model/seed/scheduler/resolution/frame workloads with and without reuse; report computed area/operations, latency, throughput, peak RAM/VRAM, GPU utilization, energy if measurable, identity drift, temporal consistency and image quality. Record unavailable metrics explicitly. Product acceptance requires actual AI video and measured reuse, not synthetic output. 10-second 1080p remains a research target.
## 2026-09-23 first working prototype gate
Latest owner criterion supersedes the earlier image/video fallback: arbitrary user text -> actual local neural VIDEO inference -> playable MP4, with recorded model/revision/settings and RAM/VRAM/time. A still image cannot satisfy this gate. First validated prompt language will be stated explicitly; Russian CLI planning is currently restricted grammar, not a general local language model. Test media and symbolic taxi integration do not pass neural acceptance. Staged weight residency must be implemented against actual backend lifetimes; ledger counters alone are insufficient. Full-world conditioning and selective compute follow as separately measured milestones.