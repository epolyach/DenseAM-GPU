# V5: Spin-Glass—Retrieval Boundary Study

## Purpose
**Direct characterization of LOCAL MINIMUM DESTRUCTION** via thermal effects in Dense Associative Memory networks.

Instead of mapping full phase diagrams (V2-V4), V5 focuses on **understanding WHY and HOW** the retrieval minimum disappears as temperature increases, with particular attention to LSR vs. LSE differences.

---

## Key Differences from V2-V4

| Aspect | V2-V4 (Phase Diagrams) | V5 (Boundary Study) |
|--------|---|---|
| **Goal** | Map full (α, T) space | Characterize minimum destruction |
| **α sampling** | 55 values (fine) | 7 values (coarse, strategic) |
| **T sampling** | 50 values (uniform) | 40 log-spaced (dense near T_c) |
| **Per-point** | Single observable (φ) | Four observables (E, φ, τ, a) |
| **Processing** | All α simultaneous | Per-α (heavy computation) |
| **Runtime** | 6-24h | ~18h (same budget, different purpose) |

---

## Physical Observables Measured

For each (α, T) pair, V5 measures **four complementary observables**:

### 1. **Free Energy Statistics** (E_mean, E_std)
- **What**: Average energy and fluctuations at thermal equilibrium
- **Why**: Detects thermodynamic instability (dF/dφ changes sign at phase transition)
- **Interpretation**: 
  - Retrieval phase: E bimodal (low φ → high E, high φ → medium E)
  - Spin-glass: E monomodal around noise floor
  - E_std jump: signature of first-order transition

### 2. **Order Parameter Distribution** (φ_mean, φ_std, φ_min, φ_max)
- **What**: Alignment with target pattern and its variance
- **Why**: Direct probe of whether retrieval is preferred
- **Interpretation**:
  - T < T_c: φ_mean ≈ 0.8-0.9, φ_std small
  - T ≈ T_c: φ_std jumps (coexistence of phases)
  - T > T_c: φ_mean ≈ 0.1-0.3, φ_std broadens

### 3. **Escape Time** (τ_esc)
- **What**: Steps to leave retrieval basin when initialized deep inside (φ₀ = 0.95)
- **Why**: Explicit measurement of barrier collapse kinetics
- **Interpretation**:
  - τ ≈ max_steps (10,000): trapped below T_c
  - τ ≈ 100-1000: barrier weakening near T_c
  - τ < 100: thermal fluctuations overcome barrier

### 4. **Acceptance Rate** (a(T))
- **What**: Fraction of accepted MC moves
- **Why**: Detects when system becomes "fluid" enough to explore spin-glass
- **Interpretation**:
  - High T: a → 50% (random walk on sphere)
  - Low T: a → 5-10% (confined to retrieval basin)
  - a(T) inflection: marks thermal destabilization

---

## Why These Four Observables?

**Complementary views of the same phenomenon:**

| Barrier Property | Observable | Signature |
|---|---|---|
| **Height** | τ_esc collapse | Exponential drop |
| **Shape** | E curve | Local minimum → inflection → gone |
| **Dynamics** | a(T) change | Acceptance increases (more accessible states) |
| **Thermodynamics** | φ distribution | Order → disorder transition |

Together, they prove: **Is the retrieval minimum destroyed by thermal effects?** (theorem → experiment)

---

## Code Structure

Both `generate_lsr_boundary_v5.jl` and `generate_lse_boundary_v5.jl` follow the same protocol:

```
For each α in {0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50}:
    
    1. Initialize patterns P(α) once
    
    2. Heavy equilibration at T_cold = 0.05 (retrieval state stable)
       - 50,000 MC steps at coldest T
    
    3. For each T in log-spaced grid:
       a. Re-equilibrate 10,000 steps (state from T-1 → T)
       b. Sample 500 steps, collect observables
       c. Measure escape time (separate run)
       d. Write row to CSV (stream, no waiting)
    
    4. Move to next α
```

**Key: Heating protocol within each α ensures no blue bay**
- State is properly equilibrated at each T (not independent)
- Thermal fluctuations naturally escape retrieval as T rises
- No artificial initialization bias

---

## Output Format

### CSV Structure
```
alpha,T,E_mean,E_std,phi_mean,phi_std,phi_min,phi_max,tau_esc,accept_rate
0.01,0.0500,-2.341,0.523,0.8923,0.0156,0.8521,0.9234,10000,0.0523
0.01,0.0561,-2.335,0.487,0.8905,0.0142,0.8601,0.9156,9876,0.0561
...
```

### Files Generated
- `lsr_boundary_v5.csv` – LSR with hard support boundaries
- `lse_boundary_v5.csv` – LSE with soft Gaussian tails (for comparison)

---

## Analysis Strategy for Paper

### 1. **Extract Critical Temperatures**
- Plot τ_esc(T) for each α → find where τ drops below ~1000
- Define T_c(α) = temperature where barrier collapses

### 2. **Compare LSR vs. LSE**
- Same plot overlaid: Why does LSR have higher T_c? (harder barriers)
- Show: LSE's τ never reaches max → no blue bay possible

### 3. **Validate Theory**
- Overlay predicted α_c(T) from saddle-point equations
- Check: Does φ_mean minimum align with theory?

### 4. **Characterize Transition Order**
- First-order: Jump in E, φ, a at single T
- Second-order: Smooth crossover
- Use φ_std peak to determine order

---

## Computational Budget (24-48h Node)

**Per α**: ~2.5 hours
- 40 T values
- 256 trials
- 50,000 init + 10,000×39 steps re-eq
- 500×40 sampling steps
- Escape time test (up to 10,000 steps)

**Total**: 7 α × 2.5 h = **17.5 hours**
- Leaves ~7h margin for repeat runs if needed
- Or run LSE in parallel (can use 2 GPUs if available)

---

## Execution

```bash
# LSR boundary study (18-24 h)
julia --project=TOML generate_lsr_boundary_v5.jl

# LSE for comparison (run on second node or sequentially)
julia --project=TOML generate_lse_boundary_v5.jl

# Analysis (Julia/Python)
# Plot τ_esc(T), E(T), φ(T), a(T) for each α
# Compare LSR vs LSE overlays
# Extract T_c(α) from τ drop
```

---

## Paper Outline (Suggested)

**"Spin-Glass—Retrieval Boundary in LSE and LSR Dense Associative Memory Networks: Monte Carlo Simulations"**

1. **Introduction**: Blue bay artifact in LSR; theoretical predictions vs. simulation
2. **Methods**: 
   - V5 design (observables, heating protocol, why it avoids blue bay)
   - LSE vs. LSR comparison rationale
3. **Results**:
   - Four observables show consistent picture of minimum destruction
   - T_c(α) extracted from τ_esc collapse
   - LSE has no blue bay (soft tails); LSR does (hard walls) — both validated
4. **Discussion**:
   - Why LSR barriers stronger: support boundary + ReLU discontinuity
   - Why heating protocol works: state evolution vs. independent initialization
   - Implications for memory capacity and retrieval reliability

---

## Key Advantages of V5 for Paper

✓ **Physically clear**: Four observables tell one story (minimum destruction)
✓ **Comparison ready**: LSR vs. LSE side-by-side with identical protocol
✓ **Theory-driven**: Directly tests saddle-point predictions about T_c(α)
✓ **Paper-quality**: Fewer α values but much deeper analysis (better visual clarity)
✓ **Novel**: No existing work does this four-observable boundary study
