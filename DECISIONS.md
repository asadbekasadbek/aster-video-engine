# Architecture decision records

ADR-000: Rust-only project implementation. Status: accepted by project owner. Native CUDA/system dependencies may be wrapped by Rust. No Python code, Python scripts, or Python inference/training service.
ADR-001: First GPU target Windows/NVIDIA; abstract backends for portability. Status: provisional until M0 research.
ADR-002: Existing licensed weights first; custom training later, with small Rust-only proof of concept. Status: provisional.
ADR-003: Evidence-based optimization and bounded resource budgets. Status: accepted.

New decisions must include date, context, alternatives, consequences, and supporting sources/benchmarks.

## ADR-004 — 2026-09-20: staged CPU baseline and first inference candidate
Status: accepted for architecture; GPU/model integration provisional pending build, approved weights and measured output.

Context: no existing implementation; Windows stable Rust/MSVC present; NVIDIA driver present but CUDA toolkit not found. 12 GiB cannot be treated as all available to model weights. See RESEARCH M0.2/M0.3 primary sources.

Decision: create a dependency-free Rust core and CLI for bounded streaming synthetic f32 latents, deterministic seed, cancellation and honest telemetry. Keep platform telemetry isolated. This is a reference computation, not pretrained inference. First model backend candidate is upstream Candle, initial model SD1.5 image smoke test (per SPEC image/video allowance). Pin release plus lockfile only after feature/build review. CUDA is first accelerated target; no automatic silent GPU-to-CPU fallback.

Alternatives: Burn is stronger for later training experiments; direct CubeCL adds kernel work without providing a video model. candle-video/LTX 2B distilled is the first separate video feasibility spike but fork dependency and custom weights license need review; Wan 1.3B is an alternative with more porting work. SVD is image-conditioned and requires temporal components absent from SD image implementation.

Gates: CUDA tensor smoke test and CPU comparison before model download; approved exact weight manifest/license; local Rust image output with peak RAM/VRAM and timing before M3 completion; separate real video output for video claim. If 12 GiB fails, reduce resolution/frame count, stage/offload components and measure; if still infeasible, retain measured image MVP and explicitly block video, never replace it with synthetic animation claims.

Consequences: no ML dependency or weight download in M1/M2. CPU baseline unblocks actual builds now. Portability is an interface goal, untested on other hardware/OS. Peak RAM uses Windows process lifetime counters; synthetic GPU utilization/VRAM/energy remain unavailable, not zero-valued invented measurements.

## ADR-005 — 2026-09-20: metadata and media boundaries
Status: implemented and locally tested on Windows.

SafeTensors 0.8.0 with existing serde dependencies is pinned after explicit user approval. Parse only a bounded header using validated upstream Metadata; independently reject repeated names and check file length. Avoid unsafe mmap and whole-weight allocation for inspection. Metadata acceptance does not claim supported model operators or payload integrity. Alternative full-buffer SafeTensors parsing would load unnecessary weight data.

Use Rust std to write bounded planar YUV420 YUV4MPEG2 streams with explicit rational frame rate, order/count validation and I/O error propagation. Use already-installed FFmpeg for independent decoding and optional MP4 preview; do not bundle its GPL-enabled binary. Synthetic motion verifies the media boundary independently of blocked pretrained inference. Sources and limitations: RESEARCH M3.1/M4.1.

## ADR-006 — 2026-09-20: isolate pinned Candle feasibility test
Status: CPU build and focused tensor/autodiff test passed; full final validation recorded in PROGRESS.

User separately authorized Candle 0.11.0 CPU dependencies. Keep the probe in aster-candle; core reference does not depend on Candle. Do not enable cuda/cudnn/mkl/metal, copy DLLs or install a toolkit implicitly. Registry Candle 0.11.0 differs from browsed main: its tokenizers dependency uses onig/native C. Existing MSVC builds it without Python. A successful CPU probe establishes only tested operations, not whole-model compatibility or video quality.
