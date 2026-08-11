# Code and data supplement

Simulation code, data, and plot scripts for the paper's Monte Carlo results on the saddle-dominated capacity of the log-sum-exp (LSE) dense associative memory.

## The three MC schemes

The paper uses three Monte Carlo schemes. Each is implemented by one script family here. The code uses internal names that differ from the paper names:

| Paper name | Code name | Script | What it does |
|---|---|---|---|
| Full | honest | `basin_stab_LSE_honest_AAAI_Nramp.jl` | All M patterns sampled explicitly per disorder. Ground truth. GPU. |
| Full (fixed N) | honest | `basin_stab_LSE_honest_AAAI.jl` | Same scheme at fixed N = 25, validation run. GPU. |
| Cone | semismart | `basin_stab_LSE_semismart_AAAI_Nramp_phikeep.jl` | Keeps only patterns with alignment above PHI_KEEP = 0.40. The discarded bulk enters as an analytic constant in the LSE log-sum. GPU. |
| Cone (fixed N) | semismart | `basin_stab_LSE_semismart_AAAI.jl` | Same scheme at fixed N = 25, validation run. GPU. |
| Bulk | kinetics | `kinetics_boundary_LSE.jl` | Metropolis chain on the analytical leading-order bulk potential. No patterns, no disorder. CPU, threaded. |

The paper names (Full, Cone, Bulk) are used everywhere in the text. The code names appear only in filenames and CSV headers.

## Layout

- `*.jl` at top level: 5 simulation scripts and 4 plot scripts.
- `data/`: the CSVs that back the paper's figures and numbers. The per-trial CSVs of the two fixed N = 25 validation runs are not included for size reasons; the scripts that produce them are (`basin_stab_LSE_honest_AAAI.jl`, `basin_stab_LSE_semismart_AAAI.jl`).
- `figures_out/`: output directory for the plot scripts, empty initially.
- `Project.toml`: Julia dependencies.

## Setup

Julia 1.9 or later. From this directory:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The plot scripts also call the external `pdfcrop` binary (part of TeX Live). Figure generation needs it on PATH.

## Regenerating the four paper figures

Each plot script reads CSVs from `data/` (if any) and writes PNG and PDF to `figures_out/`:

```bash
julia --project=. plot_cusp_vs_extreme.jl          # analytic, no data input
julia --project=. plot_cusp_illustration.jl        # analytic, no data input
julia --project=. plot_Veff_finite_N.jl            # analytic, no data input
julia --project=. plot_heatmap_textwidth_profile.jl  # reads 5 CSVs from data/
```

`plot_heatmap_textwidth_profile.jl` reads the two Cone CSVs (M = 4.4e6 and M = 3.0e8) and the Bulk boundary CSVs for N = 100, 300, 10000. The heatmap color field is the M = 3.0e8 Cone run. `plot_Veff_finite_N.jl` reads no file; it hard-codes the same N(alpha) ramp through `M_DATA = 4.4e6`.

## Re-running the simulations

Full and Cone require a CUDA GPU; the scripts assert `CUDA.functional()` at startup and exit otherwise. There is no CPU fallback. Bulk runs on CPU threads.

```bash
julia --project=. basin_stab_LSE_honest_AAAI_Nramp.jl            # Full
julia --project=. basin_stab_LSE_semismart_AAAI_Nramp_phikeep.jl # Cone
julia --project=. -t auto kinetics_boundary_LSE.jl               # Bulk
```

Conventions, shared by all simulation scripts:

- Resume is the default. Re-running continues from the last completed cell. Pass `--fresh` to overwrite the CSV and start over.
- All parameters are `const` values at the top of each file. There are no CLI flags besides `--fresh` (and `--probe` for the Cone ramp script). To change M_TARGET, PHI_KEEP, or the grids, edit the consts.
- Output CSVs are written next to the script. To feed a regenerated CSV to the heatmap plot script, move it into `data/`.
- Each CSV starts with `# generator=` comment lines recording the exact parameters of the run. Trust these over any stale header comment in a script.
- GPU memory use is auto-chunked to `MEM_BUDGET_GB`. Lower it if you hit out-of-memory errors.

The Full sweep at M = 4.4e6 took 26 days on one 48 GB GPU. The shipped `basin_stab_LSE_honest_AAAI_Nramp.csv` covers alpha = 0.20 to 0.54; the sweep continues.

## Data dictionary

`basin_stab_LSE_honest_AAAI_Nramp.csv` (Full):

| Column | Meaning |
|---|---|
| `alpha` | load, alpha = ln(M)/N |
| `T` | sampling temperature |
| `N_used` | dimension actually used for this cell |
| `disorder` | disorder realization index (1 to 32) |
| `phi_a`, `phi_b` | time-averaged alignment of replica a resp. b with the target pattern |
| `q12` | replica-replica alignment, x_a . x_b / N |
| `phi_max_other` | largest alignment with any non-target pattern |

`basin_stab_LSE_semismart_AAAI_Nramp_M*.csv` (Cone): column order is `alpha,T,N_used,K_retained,disorder,phi_a,phi_b,q12,phi_max_retained`. Two columns differ from Full:

| Column | Meaning |
|---|---|
| `K_retained` | number of patterns kept above PHI_KEEP in this disorder |
| `phi_max_retained` | largest alignment with any retained non-target pattern (replaces `phi_max_other`) |

`basin_stab_LSE_semismart_AAAI_Nramp_M*_plan.csv` (Cone budget plan, one row per alpha):

| Column | Meaning |
|---|---|
| `alpha`, `N`, `M` | load, dimension from the ramp, full pattern count |
| `K_expected` | M times P(alignment >= PHI_KEEP) under the spherical Beta((N-1)/2, (N-1)/2) density |
| `K_over_M` | retained fraction |
| `K_alloc` | allocated pattern buffer |
| `status` | `smart` if truncation is active |
| `phi_keep` | the cutoff, 0.40 |

`kinetics_boundary_LSE_N*_TMC100000.csv` (Bulk):

| Column | Meaning |
|---|---|
| `alpha` | load |
| `T_bd` | boundary temperature at which the escape criterion (threshold 0.02, window 3) first holds; `nan` above the spinodal tip |

`kinetics_boundary_LSE_N1000_TMC100000.csv` is included for completeness; the paper figure plots N = 100, 300, 10000.

## Parameters and sources

- LSE energy and the softmax inverse temperature beta_net: defined by Ramsauer et al. (2021). This paper sets beta_net = 1 throughout, matching Petrova (2026).
- MC protocol (Full and Cone): the scheme of Petrova (2026). Metropolis step size sigma = 2.4 T / sqrt(N), two replicas per disorder initialized at the target pattern, N_EQ = 32768 equilibration steps, N_SAMP = 8192 sampling steps, 32 disorder realizations per (alpha, T) cell.
- Bulk step size: curvature matched, sigma^2 = 2T / gamma(phi_eq) with gamma(phi) = T (1 + phi^2) / (1 - phi^2)^2. This replaces the 2.4 T / sqrt(N) step, which becomes microscopic at N = 1e4. Introduced in this paper. 64 chains per (alpha, T), 1e5 steps each.
- N(alpha) ramp: N(alpha) = max(12, round(ln(M_TARGET) / alpha)), round not ceil. M_TARGET = 4.4e6 (Full, and Cone at the shared budget, set by 48 GB of GPU memory at Float32) gives N = 76 to 24 over alpha = 0.20 to 0.65. M_TARGET = 3.0e8 (Cone, large budget) gives N = 98 to 30. The N_FLOOR = 12 never binds on the published range. Note: the header comment of `basin_stab_LSE_honest_AAAI_Nramp.jl` mentions 4.4e7; the `const` in the file and the CSV generator line (4.4e6) are authoritative.
- PHI_KEEP = 0.40 (Cone cutoff): introduced in this paper. The in-cone count K is a Poisson draw with mean M times the tail mass of the spherical alignment density. K stays below 1.3 percent of M in every cell.
- Grids: alpha = 0.20:0.01:0.65 and T = 0.005:0.01:0.495 in the paper. The Full script's alpha grid extends to 0.70 in code.
- Zero-temperature capacity reference alpha_c(beta_net = 1) = 0.623: Lucibello and Mezard (2024).
- Gaussian baseline boundary (grey dash-dotted curve in the heatmap): Petrova et al. (2026).

## References

- Petrova, Tatiana (2026). Thermal Robustness of Retrieval in Dense Associative Memories: LSE vs LSR Kernels. New Frontiers in Associative Memory Workshop at ICLR 2026. arXiv:2603.13350.
- Petrova, Tatiana et al. (2026). Geometric Entropy and Retrieval Phase Transitions in Continuous Thermal Dense Associative Memory. Proceedings of the 43rd International Conference on Machine Learning (ICML), PMLR. arXiv:2604.07401.
- Ramsauer, Hubert; Schäfl, Bernhard; Lehner, Johannes; Seidl, Philipp; Widrich, Michael; Gruber, Lukas; Holzleitner, Markus; Adler, Thomas; Kreil, David; Kopp, Michael K.; Klambauer, Günter; Brandstetter, Johannes; Hochreiter, Sepp (2021). Hopfield Networks is All You Need. International Conference on Learning Representations (ICLR).
- Lucibello, Carlo; Mézard, Marc (2024). The Exponential Capacity of Dense Associative Memories. Physical Review Letters 132(7), 077301.
