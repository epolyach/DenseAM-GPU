# Dense Associative Memory with Epanechnikov Energy

GPU-accelerated Monte Carlo simulations for studying phase transitions in Dense Associative Memory (DenseAM) networks, comparing Log-Sum-Exponential (LSE) and Log-Sum-ReLU (LSR) energy functions.

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

## Files

### Main Simulation Scripts
- `generate_lse_longeq_gpu.jl` - LSE energy Monte Carlo with long equilibration
- `generate_lsr_longeq_gpu.jl` - LSR energy Monte Carlo (masking version)
- `generate_lsr_longeq_gpu_v2.jl` - LSR energy Monte Carlo (T-loop version, cleaner)

### Visualization
- `maps1_longeq.jl` - Side-by-side LSE/LSR phase diagram plots

### Diagnostics
- `test_acceptance_map.jl` - Acceptance rate diagnostics across (α, T) grid
- `test_lsr_acceptance.jl` - LSR-specific acceptance rate tests

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

where `c ≈ 0.15` maintains approximately constant accepted moves across temperatures.

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

Key difference: LSR has finite support regions (cones) around each pattern, creating hard boundaries and allowing emergent local minima at cone intersections.

## Output

- Phase diagrams showing φ = ⟨target·x⟩/N across (α, T) plane
- Blue regions: retrieval phase (φ ≈ 1)
- Red regions: spin-glass phase (φ ≈ 0)
- Black curves: theoretical phase boundaries

## Requirements

- Julia 1.9+
- CUDA.jl (GPU support)
- Plots.jl, CSV.jl, DataFrames.jl
- NVIDIA GPU (tested on A6000)

## References

- Hoover et al., "Dense Associative Memory with Epanechnikov Energy", ICLR 2025 Workshop
- Lucibello & Mézard, "Exponential capacity of dense associative memories", PRL 2024
