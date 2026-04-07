# Basin Percolation in LSR Associative Memory

The retrieval-to-paramagnetic transition in the Log-Sum-ReLU (LSR) Dense Associative Memory corresponds to **percolation of pattern basins** on the sphere S^{N-1}(sqrt(N)). Each pattern's basin is a spherical cap with a hard energy wall at overlap phi_c = (b-1)/b. When the load alpha = ln(P)/N exceeds a critical threshold, neighboring caps overlap and form a giant connected cluster — the system can no longer confine dynamics to a single target.

## Key results

- **Percolation threshold**: alpha_c = phi_c^2 / 2 = 1/4 (at b = 2+sqrt(2), N -> infinity), with a finite-N correction ~ 0.083/sqrt(N)
- **Poisson model**: the neighbor count K is well-approximated by Poisson(lambda), validated by direct MC sampling
- **Branching process**: survival probability shows a sharp transition at alpha_c, confirmed at multiple BFS depths
- **b-scaling**: as b grows from O(1) toward O(N), phi_c -> 1, alpha_c shifts from 1/4 toward 1/2, and energy barriers drop from O(N) to O(1)

## Repository structure

```
code/
  percolation_LSR.jl             # Main 4-panel analysis (lambda, K vs Poisson, survival, BFS)
  percolation_bscaling.jl        # b-scaling: alpha_c(b), barriers, validation across b values
  basin_connectivity_schematic.jl # Schematic: isolated basins (low T) vs giant cluster (high T)
  basin_connectivity_schematic.m  # MATLAB version of the schematic

data/
  percolation_LSR_panel2.csv     # Neighbor count K vs expected lambda
  percolation_LSR_panel3.csv     # Branching process survival at depths 1, 3, 10, 50
  percolation_LSR_panel4.csv     # BFS depth-1 vs depth-2 comparison
  percolation_bscaling.csv       # K and lambda across multiple b values

figures/
  percolation_LSR.png/pdf        # Main 4-panel figure
  percolation_bscaling.png/pdf   # b-scaling 4-panel figure
  basin_connectivity_schematic.png

latex/
  percolation_LSR.tex            # Manuscript source
  percolation_LSR.pdf            # Compiled manuscript
  percolation_LSR.png            # Figure copy for LaTeX compilation
  percolation_bscaling.png       # Figure copy for LaTeX compilation

papers/                          # Reference papers
```

## Usage

```bash
# Regenerate percolation data and figures
julia percolation_LSR.jl --refresh_csv
julia percolation_bscaling.jl --refresh_csv

# Plot from cached CSV (no recomputation)
julia percolation_LSR.jl
julia percolation_bscaling.jl

# Compile manuscript
cd latex && pdflatex percolation_LSR.tex
```

## Dependencies

Julia packages: Random, LinearAlgebra, Statistics, Printf, Plots, DelimitedFiles

## References

1. Hoover et al., "Dense Associative Memory with Epanechnikov Energy", ICLR 2025 Workshop
2. Lucibello & Mezard, "Exponential capacity of dense associative memories", PRL 132, 077301 (2024)
