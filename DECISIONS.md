# Architecture decision records

ADR-000: Rust-only project implementation. Status: accepted by project owner. Native CUDA/system dependencies may be wrapped by Rust. No Python code, Python scripts, or Python inference/training service.
ADR-001: First GPU target Windows/NVIDIA; abstract backends for portability. Status: provisional until M0 research.
ADR-002: Existing licensed weights first; custom training later, with small Rust-only proof of concept. Status: provisional.
ADR-003: Evidence-based optimization and bounded resource budgets. Status: accepted.

New decisions must include date, context, alternatives, consequences, and supporting sources/benchmarks.
