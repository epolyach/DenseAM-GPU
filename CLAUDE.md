# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Writing style (use for all discussion)

Plain, direct, reserved. Short sentences. Numbers before adjectives.
No "drags", "collapses toward", "vanishingly", "It turns out that", "Note that".
One claim per sentence. No restating the question. No closing summary.
Math is fine when it carries content; prose around it stays skeletal.

## Repo layout: active vs archive

The repo holds three generations of work; only the newest is currently being developed.

- **`AAAI_Ramsauer/` + `LSE_capacity_final/`** — **active**. The AAAI 2026 project on the saddle-dominated capacity of the LSE model: simulations and plot scripts in `AAAI_Ramsauer/`, the submission LaTeX in `LSE_capacity_final/`. All recent commits touch these trees. (`LSE_capacity/` is an earlier copy of the same paper, kept with its build dirs; `aaai2026_LSE_saddle*.tex` and `LSE_capacity_v1/v2` inside `AAAI_Ramsauer/` are superseded drafts. Edit `LSE_capacity_final/` only.)
- **`NIPS_Resilience/code/`** — previous generation. Smart-MC / honest-MC basin-stability studies for LSE and LSR, plus their plot scripts. Still the canonical location for the smart-MC and LSR work.
- **Repo root** (`basin_stab_LSE_v6.jl`, `generate_lsr_longeq_gpu_v4.jl`, etc.) — **archive**, last touched March 2025. The README's v1→v4 LSR evolution narrative is historical; do not assume those scripts represent the current methodology.
- **`OLD_*`, `NeurIPS_2024/`, `PROD*`, `panels_v10/`** — frozen artifacts. Read-only for reference.

When the user says "the paper" without qualification, they mean `LSE_capacity_final/`. "The smart-MC code" means `NIPS_Resilience/code/`; scripts for the AAAI project live in `AAAI_Ramsauer/`.

## AAAI 2026 paper (`LSE_capacity_final/`)

Three tex targets, synced by hand (no `\input` sharing):

| File | Content |
|---|---|
| `LSE_capacity.tex` | Joint compile: main text + Appendices A-C. The reference version. |
| `LSE_capacity_na.tex` | Main text only (page-limited submission body). |
| `LSE_capacity_appendices.tex` | Standalone appendices (separate technical-appendix upload). |

Sync rules:
- A main-text edit goes into both `LSE_capacity.tex` and `LSE_capacity_na.tex`. The two are line-for-line identical except: `_na` writes appendix cross-references as literal `Appendix~A` / `Appendix~B` instead of `\ref{...}`, and stops at `\end{document}` after `\bibliography`.
- An appendix edit goes into both `LSE_capacity.tex` and `LSE_capacity_appendices.tex`.
- Equation numbers (36)-(53) in `LSE_capacity_appendices.tex` are frozen with `\tag{N}` to match the joint compile; its references to main-text equations are hard-coded numerals. Any edit that shifts equation numbering in the joint compile requires re-freezing those tags and numerals by hand.

The preamble is the official AAAI 2026 template; lines marked `DO NOT CHANGE` are literal, and the forbidden package/command list sits in comments at the top of each tex file. Both main files use `\usepackage[submission]{aaai2026}` (anonymous).

Build from `LSE_capacity_final/` so `panels_paper/` figure paths resolve; send aux output to a per-target build dir (convention from `LSE_capacity/`):

```bash
mkdir -p build_joint
pdflatex -interaction=nonstopmode -output-directory=build_joint LSE_capacity.tex
cp aaai2026.bib aaai2026.bst build_joint/
(cd build_joint && bibtex LSE_capacity)
pdflatex -interaction=nonstopmode -output-directory=build_joint LSE_capacity.tex
pdflatex -interaction=nonstopmode -output-directory=build_joint LSE_capacity.tex
```

Same recipe with `build_na` / `LSE_capacity_na` and `build_app` / `LSE_capacity_appendices`. Keep `.aux`/`.log` files out of the source directory.

`LSE_capacity_final/panels_paper/*.pdf` are copies of `AAAI_Ramsauer/panels_paper/` outputs. To change a figure, edit and run the producing script in `AAAI_Ramsauer/` (Julia + CairoMakie; each script reads CSVs next to itself and states its output in a header comment), then copy the new PDF over:

| Figure | Producing script |
|---|---|
| `cusp_vs_extreme.pdf` | `plot_cusp_vs_extreme.jl` |
| `cusp_illustration.pdf` | `plot_cusp_illustration.jl` |
| `heatmap_textwidth_profile.pdf` | `plot_heatmap_textwidth_profile.jl` |
| `Veff_finite_N.pdf` | `plot_Veff_finite_N.jl` |

Naming: the paper's three MC schemes map to `AAAI_Ramsauer/` scripts as Full = `basin_stab_LSE_honest_AAAI*.jl` ("honest" in filenames/CSVs), Cone = `basin_stab_LSE_semismart_AAAI*.jl` ("semismart"), Bulk = `kinetics_boundary_LSE.jl` / `kinetics_LSE_Ndim.jl` (chain on the analytical bulk potential). Use only the paper names (Full, Cone, Bulk) in the tex; the code names never appear in prose.

Run facts behind the paper's numbers (from script headers and CSV comment lines):
- `N(α) = max(N_FLOOR, round(ln M_TARGET / α))` — round, not ceil. `M_TARGET = 4.4e6` (Full, and Cone at the shared budget) gives N = 76…24 over α = 0.20…0.65; `M_TARGET = 3.0e8` (Cone, `PHI_KEEP = 0.40`) gives N = 98…30.
- Full/Cone are GPU scripts with resume-by-default and `--fresh`, like the NIPS tree. Bulk (`kinetics_boundary_LSE.jl`) is CPU: `julia -t auto`, output `kinetics_boundary_LSE_N{N}_TMC{T_MC}.csv`, also resume/`--fresh`.
- CSVs carry a `# generator=…` comment header with the exact parameters; trust it over the paper text when they disagree.

`LSE_capacity_final/Backups/` holds manual snapshots named `<file>_backup_vN-MM-DD-HHhMM.tex`. When asked to back up before an edit pass, follow that naming; do not overwrite existing snapshots.

## Canonical scripts in `NIPS_Resilience/code/` (previous generation)

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
