# Aster Video Engine — Rust-only research project

Goal: build a modular, measurable, cross-platform AI video-generation engine. Development host: Windows, RTX 4070 12 GB VRAM, 64 GB RAM. Rust is the only project implementation language; native GPU drivers, system libraries and model files may be external dependencies. No Python scripts, Python inference servers or Python training pipelines.

## Start here
1. Read `AGENTS.md`, `SPEC.md`, `ROADMAP.md`, `TASKS.md`, `PROGRESS.md`, `DECISIONS.md`.
2. Inspect the host with safe read-only commands; check `rustc`, `cargo`, `git`, `nvidia-smi`, free disk and available GPU backends.
3. Research current Rust ML options and model licenses using primary documentation; log links and dates in `RESEARCH.md`.
4. Implement the first uncompleted acceptance-tested milestone, not the entire vision at once.
5. Build, test, benchmark where applicable, review, document and checkpoint progress. Repeat.

## First run
In this folder: `git init`, then `opencode`. Send: `Read AGENTS.md and all referenced project files. Execute the first unfinished milestone. Keep iterating within permissions, verify each change, and update PROGRESS.md.`

## Success is evidence-based
An empty skeleton is not a video generator. A complete engine must pass the acceptance tests in SPEC.md. Claims of 1080p, speed, quality, or training are prohibited without reproducible measurements and artifacts.
