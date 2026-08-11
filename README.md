# Dense Associative Memory at Finite Temperature

GPU Monte Carlo and theory for retrieval in continuous Dense Associative Memories (DAMs) on the N-sphere, as a function of the load α = ln(M)/N and the bath temperature T. Two kernel energies are compared throughout: log-sum-exp (LSE, Gaussian kernel) and log-sum-ReLU (LSR, Epanechnikov kernel, compact support). The repository holds the code, data, and LaTeX behind four papers, written in that order; each generation of code supersedes the previous one.

## Papers and their trees

| Paper | Status | Trees |
|---|---|---|
| Finite-T retrieval phase diagrams for LSE and LSR | ICML 2026 | `icml2026/` (paper); simulations from the equilibrium generation: root scripts, `PROD/`, `PROD_V3/`, `panels_v10/`, `CSV/` |
| Thermal robustness of LSE vs LSR (direct MC) | ICLR 2026 workshop | same equilibrium generation (root `generate_*`, `maps_*`, `basin_stab_*_v6/v7`) |
| LSR metastability: Kramers escape and centroid hopping | under review | `NIPS_Resilience/` (code, `latex/metastable.tex`, `panels_paper/`) |
| LSE saddle-dominated capacity | submitted | `AAAI_Ramsauer/` (simulations, plot scripts), `LSE_capacity_final/` (paper + code supplement); `LSE_capacity/` is an earlier copy |

## Repository layout

Active and recent:

- `AAAI_Ramsauer/` — LSE capacity simulations. Three MC schemes: Full (`basin_stab_LSE_honest_AAAI*.jl`, all M patterns), Cone (`basin_stab_LSE_semismart_AAAI*.jl`, explicit patterns inside a cone around the target, analytical bulk outside), Bulk (`kinetics_boundary_LSE.jl`, `kinetics_LSE_Ndim.jl`, chain on the analytical bulk potential, CPU). CSVs sit next to the scripts; plots go to `AAAI_Ramsauer/panels_paper/`.
- `LSE_capacity_final/` — the LSE capacity paper: three tex targets (joint, main-only, appendices), figures in `panels_paper/`, code supplement in `code_supplement/`.
- `NIPS_Resilience/` — the LSR metastability study. `code/` holds the smart-MC and honest-MC scripts (latest: `basin_stab_LSE_smart_v19_dynamic.jl`, `basin_stab_LSE_honest_fixedN.jl`, `basin_stab_LSR_v17.jl`) and the plot scripts; `latex/` the paper and notes; `panels_paper/` the figures.
- `icml2026/` — the ICML paper source and revisions.

Archive (read-only for reference):

- Root scripts (`generate_lsr_longeq_gpu*.jl`, `generate_lse_longeq_gpu.jl`, `basin_stab_*_v6/v7.jl`, `maps_*.jl`, `trajectory_*.jl`, `percolation_*.jl`) — the equilibrium generation, last touched March 2025. The v1→v4 evolution of the LSR runs is documented in `LATEX/lsr_evolution.tex`; a summary is in the History section below.
- `NIPS_Percolation/` — percolation side study of the LSR support graph.
- `PROD/`, `PROD_V3/`, `panels_v10/`, `panels_paper/`, `CSV/`, `EPS/`, `PNG/`, `OLD_*` — frozen production runs and figure sets.
- `MD/GPU_OPTIMIZATION_GUIDE.md` — alpha-batching, fused ops, preallocation. Written for the archive scripts; the principles still apply.

## Running

CUDA is mandatory for the `basin_stab_*` and `generate_*` GPU scripts; they assert `CUDA.functional()` and exit otherwise. The Bulk scheme (`kinetics_boundary_LSE.jl`) is CPU (`julia -t auto`).

```bash
# Archive root scripts: use the TOML/ project
julia --project=TOML -e 'using Pkg; Pkg.instantiate()'   # first time
julia --project=TOML <script>.jl

# NIPS_Resilience and AAAI_Ramsauer scripts: run from their directory
cd NIPS_Resilience/code
julia <script>.jl              # resume from the last completed (α, T)
julia <script>.jl --fresh      # start over, overwrite the CSV
```

Conventions shared by the recent generations: all parameters are `const` blocks at the top of each script (no CLI flags besides `--fresh`); resume is the default; GPU memory is auto-chunked to `TARGET_MEM_PER_CHUNK_GB`; CSVs carry a `# generator=...` header with the exact run parameters.

## Model

Patterns and state live on the sphere |x|² = N; the number of patterns is exponential in the size, M = e^{αN}. The alignment with pattern ξ_μ is φ_μ = x·ξ_μ/N.

- LSE: E = −(1/β) ln Σ_μ exp(−β/2 ||x − ξ_μ||²). Every pattern contributes at every alignment.
- LSR: E = −(1/β) ln Σ_μ ReLU(1 − β/2 ||x − ξ_μ||²). A pattern contributes only inside its support cap; at b = 2 + √2 the support wall sits at φ_c = (b−1)/b = 1/√2, and the geometric threshold is α_th = −½ ln(1 − φ_c²) = ½ ln 2 ≈ 0.347.

The finite-T program: equilibrium phase diagrams in (α, T) for both kernels (ICML, workshop), then the escape kinetics behind the apparent boundaries. In LSR the state escapes through an individual spurious pattern over a purely entropic barrier and hops between two-pattern centroids; in LSE it escapes collectively into a basin formed jointly by many weakly aligned patterns near an interior saddle. The apparent finite-N boundary is a kinetic crossover at τ ≈ T_MC, not a thermodynamic transition.

## History: the LSR equilibrium runs, v1→v4

The root-level LSR simulation evolved through four versions: v1 batched all T with an active mask; v2 ran a sequential T loop with per-(α,T) states and showed a "blue bay" artifact (metastable retrieval above the boundary from independent initialization); v3 introduced the heating protocol (equilibrated states propagate from low to high T), which removed the artifact; v4 added CUDA streams and double-buffered RNG on a finer grid. Details in `LATEX/lsr_evolution.tex`. The methodology of the later generations (confirmed-barrier-crossing escape criterion, smart-MC live sets, the three AAAI schemes) supersedes this narrative.

## Requirements

Julia 1.9+, CUDA.jl, CairoMakie/Plots, CSV.jl, DataFrames.jl; an NVIDIA GPU (tested on RTX A6000, 48 GB). `NIPS_Resilience/code/Project.toml` is minimal; if a package is missing there, run with `--project=..`.

## References

1. Hoover et al., "Dense Associative Memory with Epanechnikov Energy", ICLR 2025 Workshop (`papers/25_Dense_Associative_Memory_wi.pdf`)
2. Lucibello, Mézard, "Exponential capacity of dense associative memories", PRL 132, 077301 (2024)
3. Ramsauer et al., "Hopfield Networks is All You Need", ICLR 2021
4. Krotov, Hopfield, "Dense Associative Memory for Pattern Recognition", NeurIPS 2016
5. Demircigil et al., "On a model of associative memory with huge storage capacity", J. Stat. Phys. 168, 288 (2017)
