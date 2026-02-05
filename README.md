# Dense Associative Memory with Epanechnikov Energy

GPU-accelerated Monte Carlo simulations for studying phase transitions in Dense Associative Memory (DenseAM) networks, comparing Log-Sum-Exponential (LSE) and Log-Sum-ReLU (LSR) energy functions.

Based on: [Hoover et al., "Dense Associative Memory with Epanechnikov Energy", ICLR 2025 Workshop](papers/25_Dense_Associative_Memory_wi.pdf)

## Overview

This repository contains Julia code for:
- GPU-accelerated Metropolis-Hastings Monte Carlo on the N-sphere
- Phase boundary detection for LSE and LSR energy functions
- T-dependent equilibration for proper sampling at low temperatures
- Visualization of phase diagrams (retrieval vs spin-glass phases)

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `b` | LSR sharpness (Epanechnikov kernel) | 2 + √2 ≈ 3.414 |
| `α = ln(P)/N` | Memory load | 0.01 - 0.55 |
| `T` | Temperature | 0.05 - 2.50 |
| `N_TRIALS` | Independent MC chains | 64 - 256 |
| `N_EQ` | Equilibration steps | T-dependent |
| `N_SAMP` | Sampling steps | 500 - 5000 |

## LSR Simulation Versions

The LSR Monte Carlo simulation evolved through four versions, each addressing specific issues in the phase diagram:

### v1 — Masking (`generate_lsr_longeq_gpu.jl`)
- Processes all T simultaneously in batched arrays `[N, n_T×N_TRIALS]`
- Uses `mc_step_masked!` with boolean active mask for T-dependent equilibration
- All chains initialized independently near target for every (α, T)
- **Problem**: Complex threshold-based deactivation; all states initialized in retrieval basin

### v2 — T-loop (`generate_lsr_longeq_gpu_v2.jl`)
- Sequential T loop with per-(α, T) state arrays `xs_full[i][j]`
- T-dependent equilibration: `N_EQ(T) = 5000 × exp(0.15 × (1/T - 1/T_max))`, cap 300k
- Uses `gemm_strided_batched` for batched energy computation
- **Problem**: "Blue bay" artifact — metastable retrieval above the theoretical boundary due to independent initialization + first-order transition barrier + LSR hard support walls

### v3 — Heating Protocol (`generate_lsr_longeq_gpu_v3.jl`)
- **Key innovation**: propagates equilibrated states from low T → high T (heating protocol)
- Single state per α `xs_g[i]` reused across T — massive memory reduction
- Only T₁ initialized near target; subsequent T inherit state from previous T
- Merged equilibration + sampling in one T loop
- Log-spaced T grid for better resolution at low T where the phase transition occurs
- **Result**: Blue bay eliminated — thermal fluctuations naturally destabilize retrieval as T crosses T_c

### v4 — CUDA Streams + Fine Grid (`generate_lsr_longeq_gpu_v4.jl`) ← current
- **CUDA streams**: one stream per α value for concurrent GPU processing
- **Double-buffered RNG**: overlap random number generation with compute (one `CUDA.synchronize()` per MC step)
- `mc_step_prerand!`: separates random generation from compute (cuRAND generators can't be shared across concurrent streams)
- Finer grids: α = 0.01:0.01:0.55 (55 values), T = 50 log-spaced points
- Heating protocol from v3 preserved
- Output: `lsr_heating.csv`

See [LATEX/lsr_evolution.tex](LATEX/lsr_evolution.tex) for detailed documentation of the v1→v2→v3 evolution.

## Project Structure

```
├── generate_lse_longeq_gpu.jl    # LSE Monte Carlo simulation
├── generate_lsr_longeq_gpu.jl    # LSR v1: masking approach
├── generate_lsr_longeq_gpu_v2.jl # LSR v2: T-loop, independent states
├── generate_lsr_longeq_gpu_v3.jl # LSR v3: heating protocol
├── generate_lsr_longeq_gpu_v4.jl # LSR v4: CUDA streams + fine grid (current)
├── maps1_longeq.jl               # Phase diagram visualization
├── test_acceptance_map.jl        # Acceptance rate diagnostics
├── test_lsr_acceptance.jl        # LSR acceptance tests
├── TOML/
│   └── Project.toml              # Julia dependencies
├── LATEX/
│   ├── lsr_evolution.tex         # LSR v1→v2→v3 evolution documentation
│   ├── fss_report.tex            # Finite-size scaling report
│   └── mc_approaches.tex         # MC methodology notes
├── MD/
│   ├── README.md                 # Usage notes
│   └── GPU_OPTIMIZATION_GUIDE.md # GPU optimization tips
└── papers/
    └── 25_Dense_Associative_Memory_wi.pdf  # Reference paper
```

## Usage

```bash
# First time setup
julia --project=TOML -e 'using Pkg; Pkg.instantiate()'

# Run LSE simulation
julia --project=TOML generate_lse_longeq_gpu.jl

# Run LSR simulation (v4 recommended)
julia --project=TOML generate_lsr_longeq_gpu_v4.jl

# Generate phase diagram plots
julia --project=TOML maps1_longeq.jl
```

## Equilibration Strategies

### v2: T-dependent exponential
```
N_EQ(T) = N_EQ_base × exp(c × (1/T - 1/T_max)),  cap 300k
```
Compensates for exponentially decreasing acceptance rate at low T.

### v3/v4: Heating protocol
```
N_EQ_INIT at T₁ (coldest) + N_EQ_STEP per subsequent T step
```
State propagates from low T → high T. At T < T_c: system stays in retrieval. At T ≈ T_c: thermal fluctuations naturally cross the weakening barrier. At T > T_c: system has already transitioned to spin-glass.

## Physics Background

### LSE Energy (Log-Sum-Exponential)
Standard modern Hopfield network energy with Gaussian kernel:
```
E_LSE(x) = -1/β × log Σ_μ exp(-β/2 × ||x - ξ_μ||²)
```

### LSR Energy (Log-Sum-ReLU)
Novel energy function based on Epanechnikov kernel:
```
E_LSR(x) = -1/β × log Σ_μ ReLU(1 - β/2 × ||x - ξ_μ||²)
```

### Key Differences

| Property | LSE | LSR |
|----------|-----|-----|
| Support | Infinite (Gaussian tails) | Finite (cones with θ ≈ 45°) |
| Boundary | Soft | Hard (infinite energy barrier) |
| Emergent minima | None | At cone intersections |
| Support boundary | N/A | φ = 1 - 1/b ≈ 0.707 |

### Phase Diagram Interpretation

- **Blue regions** (φ ≈ 1): Retrieval phase - system recalls target pattern
- **Red regions** (φ ≈ 0): Spin-glass phase - system trapped in spurious states
- **Theoretical boundary**: Black curve separating phases

For LSR, the phase boundary includes:
- Curved portion: `α_c(T)` from saddle-point equations
- Vertical portion at `α_th = 0.5 × (1 - 1/b)² ≈ 0.043`

## Output Files

Running simulations generates:
- `lse_longeq.csv` / `lsr_longeq.csv` - Raw φ data across (α, T) grid
- `maps1_longeq.png` - Side-by-side phase diagrams

## Requirements

- Julia 1.9+
- CUDA.jl (GPU support)
- Plots.jl, CSV.jl, DataFrames.jl, ProgressMeter.jl
- NVIDIA GPU (tested on RTX A6000, 48GB)

## References

1. Hoover et al., "Dense Associative Memory with Epanechnikov Energy", ICLR 2025 Workshop
2. Lucibello & Mézard, "Exponential capacity of dense associative memories", PRL 132, 077301 (2024)
3. Ramsauer et al., "Hopfield Networks is All You Need", ICLR 2021
