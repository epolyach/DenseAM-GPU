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

## Project Structure

```
├── generate_lse_longeq_gpu.jl    # LSE Monte Carlo simulation
├── generate_lsr_longeq_gpu.jl    # LSR Monte Carlo (masking version)
├── generate_lsr_longeq_gpu_v2.jl # LSR Monte Carlo (T-loop version, recommended)
├── maps1_longeq.jl               # Phase diagram visualization
├── test_acceptance_map.jl        # Acceptance rate diagnostics
├── test_lsr_acceptance.jl        # LSR acceptance tests
├── TOML/
│   └── Project.toml              # Julia dependencies
├── LATEX/
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

# Run LSR simulation (v2 recommended)
julia --project=TOML generate_lsr_longeq_gpu_v2.jl

# Generate phase diagram plots
julia --project=TOML maps1_longeq.jl
```

## T-dependent Equilibration

For proper equilibration at low T, the number of equilibration steps scales as:

```
N_EQ(T) = N_EQ_base × exp(c × (1/T - 1/T_max))
```

where `c ≈ 0.15` maintains approximately constant accepted moves across temperatures. This compensates for the exponentially decreasing acceptance rate at low T.

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
