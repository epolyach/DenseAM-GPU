# Algorithm Comparison: Julia vs MATLAB for LSE at α=0.1

## Energy Formula (IDENTICAL)

Both use the same LSE energy:
```
E = -ln(Σ exp(-N*(1-φ)))
```
with logsumexp numerical stabilization.

## Key Parameter Differences

| Parameter | MATLAB (quick_validation_5.m) | Julia (fast) | Julia (default) |
|-----------|-------------------------------|--------------|-----------------|
| **N** | **50** | **30** ⚠️ | **50** ✓ |
| **P at α=0.1** | exp(5) ≈ **148** | exp(3) ≈ **20** ⚠️ | exp(5) ≈ **148** ✓ |
| Initial noise | 0.05 | **0.1** ⚠️ | **0.1** ⚠️ |
| Step size | 0.2 | **0.25** | **0.25** |
| n_eq | 1500 | 1000 | 2000 ✓ |
| n_samp | 1000 | 800 | 1500 ✓ |
| n_trials | 1 | 2 | 3 |

## Critical Differences Affecting Results

### 1. **Network Size N (MOST IMPORTANT)**
- **MATLAB**: N = 50 → P = exp(0.1×50) ≈ 148 patterns
- **Julia fast**: N = 30 → P = exp(0.1×30) ≈ 20 patterns ⚠️

**Impact**: With N=30, the physics is fundamentally different:
- Energy scales as E ∝ N
- Fewer patterns (20 vs 148) → less interference
- Phase transition occurs at different temperatures

### 2. **Initialization Noise**
```matlab
% MATLAB:
x = target + 0.05 * randn(N, 1);  % 5% noise
```
```julia
# Julia:
x .= target .+ F(0.1) .* CUDA.randn(...)  # 10% noise (2× larger)
```

**Impact**: Larger initial noise may require more equilibration steps.

### 3. **Monte Carlo Parameters**
- Different equilibration steps affect convergence
- Julia runs multiple trials (2-3) and averages
