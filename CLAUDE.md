# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Writing style (use for all discussion)

Plain, direct, reserved. Short sentences. Numbers before adjectives.
No "drags", "collapses toward", "vanishingly", "It turns out that", "Note that".
One claim per sentence. No restating the question. No closing summary.
Math is fine when it carries content; prose around it stays skeletal.

## Repo layout: active vs archive

The repo holds two generations of work; only one is currently being developed.

- **`NIPS_Resilience/code/`** — **active**. All recent commits touch this tree. Smart-MC / honest-MC basin-stability studies for LSE and LSR live here, plus their plot scripts.
- **Repo root** (`basin_stab_LSE_v6.jl`, `generate_lsr_longeq_gpu_v4.jl`, etc.) — **archive**, last touched March 2025. The README's v1→v4 LSR evolution narrative is historical; do not assume those scripts represent the current methodology.
- **`OLD_*`, `NeurIPS_2024/`, `PROD*`, `panels_v10/`** — frozen artifacts. Read-only for reference.

When the user says "the LSE script" or "the smart-MC code" without qualification, they mean files in `NIPS_Resilience/code/`.

## Current canonical scripts (in `NIPS_Resilience/code/`)

| Topic | Latest file | Notes |
|---|---|---|
| Smart-MC LSE | `basin_stab_LSE_smart_v19_dynamic.jl` | Dynamic live-set regeneration every `N_REFRESH` MC steps. Supersedes v18 (static live set). |
| Honest-MC LSE | `basin_stab_LSE_honest_fixedN.jl` | Ground-truth reference at small N (N=25) for boundary validation. |
| LSR | `basin_stab_LSR_v17.jl` | Escape-time / D_v zone classification. v16, v16b are alternates. |
| Heatmaps | `plot_LSE_smart_v19_heatmap.jl`, `plot_heatmap_3phase.jl`, `plot_section3_heatmap.jl` | Read CSV next to script, write PDF/PNG to `../panels_paper/`. |

Versioned-script suffix convention:
- bare `vN` — incremental algorithm refinement
- `vNm`, `vNb` — variant of vN (modified / sub-variant)
- `vN_<word>` — specialized focus (`v11_escape`, `v12_Nscaling`, `v14_tau`, `v15_ventry`)
- `smart_vN` — budget-truncated K-pattern live set with passive-sea constant; the rest are honest MC over all M patterns
- `_dynamic`, `_a1`, etc. — sub-flavors (refresh policy, initial overlap, …)

## Running things

Two run patterns coexist; pick by directory:

```bash
# Root-level archive scripts (use the TOML/ project)
julia --project=TOML <script>.jl

# Active NIPS_Resilience scripts — run from NIPS_Resilience/code/
cd NIPS_Resilience/code
julia <script>.jl              # resume from last completed (α, T)
julia <script>.jl --fresh      # start over, overwriting the CSV
```

First-time dep install:
```bash
julia --project=TOML -e 'using Pkg; Pkg.instantiate()'
```

These are research scripts: no test suite, no lint/format step, no build target. "Run a test" means launching a simulation and checking its CSV/plot output.

## Outputs

- **CSV** — written next to the script that produced it (e.g. `basin_stab_LSE_smart_v19_N500_K10000.csv` lives in `NIPS_Resilience/code/`). Filename encodes the key params (`N`, `K`, sometimes `α`/`T`).
- **Plots** — `plot_*.jl` scripts in `NIPS_Resilience/code/` write to `NIPS_Resilience/panels_paper/` via `joinpath(@__DIR__, "..", "panels_paper")`. Output is overwritten silently. (There is also a `NIPS_Resilience/code/panels_paper/` from one script — prefer the parent-level one unless the script clearly targets the local copy.)

## Things to know before editing

1. **CUDA is mandatory.** All `basin_stab_*` scripts assert `CUDA.functional()` at startup and exit otherwise. There is no CPU fallback.
2. **All parameters are `const` at the top of the file.** No CLI flags besides `--fresh`. To change `N`, `K`, the α grid, the T grid, or memory budget, edit the `const` block and re-run. Do not refactor these into kwargs without being asked.
3. **Resume is the default behavior** — re-running a script without `--fresh` picks up where the last run stopped. Useful, but means a silently broken run can leave a CSV that looks complete; check the row count against the α×T grid before trusting it.
4. **GPU memory is auto-chunked** to a target (`TARGET_MEM_PER_CHUNK_GB`, often 40 GB; LSR scripts may use 5 GB). Lower it if you OOM; raise it for speed on big cards.
5. **T grids are α-dependent, not rectangular.** Smart-MC scripts compute a per-α T range from `φ_cut(α, K)` and a `T_safety` factor. Don't assume a fixed T axis when joining CSVs across α.
6. **`NIPS_Resilience/code/Project.toml` is minimal** (only `Distributions`). It does not pin CUDA / CairoMakie versions — those come from the global env or the parent `NIPS_Resilience/Project.toml`. If a script fails to find a package, run it with `--project=..` from `NIPS_Resilience/code/`.

## Documentation pointers

- `README.md` (root) — physics background, parameter table, the v1→v4 LSR story.
- `NIPS_Resilience/latex/` — the live papers and notes (`metastable.tex`, `note_LSE_fss.tex`, `smart_MC_validity.tex`, `resilience_dop.tex`). When the user asks about methodology motivation, look here before guessing.
- `MD/GPU_OPTIMIZATION_GUIDE.md` — alpha-batching / fused-op / preallocation tips. Written for the archive scripts but the principles still apply.
