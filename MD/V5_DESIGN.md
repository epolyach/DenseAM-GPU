# V5: Spin-Glass Boundary Characterization via Local Minimum Destruction

## Empirical Knowledge Consolidated from V1-V4

### Pattern & Dimension Scaling
```
N(α) = ⌊ln(P) / α⌋  (adaptive)
P(α) = MIN_PAT + slope × (α - α_min)

V2/V3 Parameters:
- MIN_PAT = 500, MAX_PAT = 20000
- LSE: MIN_PAT = 200, MAX_PAT = 5000 (less demanding)
- LSR requires ~2.5× more patterns for numerical stability
```

### Equilibration Lessons

**LSE (works well):**
- Flat N_EQ = 50,000 steps for all T
- No meta-stability issues (soft Gaussian tails)
- N_TRIALS = 256, N_SAMP = 5,000 sufficient

**LSR (problematic):**
- Hard support walls create barriers
- T-dependent equilibration necessary
- V3 heating protocol eliminates blue bay (key insight)
- V2 approach fails: independent initialization → metastability

### Tested Step Sizes
```julia
δ(N) = max(0.1, 2.4/√N)  # Robust adaptive scaling
      # Acceptance rates: ~30-50% at high T, 5-10% at low T
```

### Known Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Blue bay | Independent (α,T) init + hard walls | Heating protocol (V3) |
| Wasted compute | Masking inactive chains | T-loop (V2) or heating (V3) |
| GPU parallelization limits | Sequential α processing | CUDA streams (V4) |

---

## V5 Purpose: Understand LOCAL MINIMUM DESTRUCTION Mechanism

**Question**: What T destroys the retrieval local minimum? How does this depend on α, N, P?

### Observables to Measure

1. **Free Energy of Retrieval State** (direct)
   - E_ret(T) = ⟨E⟩ at equilibrium starting near target
   - Derivatives: dE/dT, d²E/dT²

2. **Escape Time from Retrieval Basin** 
   - τ_esc = # steps to reach φ < threshold
   - Initialized at φ_init = 0.95 (inside basin)
   - Measures barrier collapse: τ → ∞ below T_c, τ finite above

3. **Order Parameter Statistics**
   - φ_mean, φ_std: estimates of order at equilibrium
   - φ_min, φ_max: support of φ distribution

4. **Acceptance Rate Temperature Dependence**
   - a(T) = acceptance rate
   - Rapid change near T_c (signature of phase transition)

---

## V5 Code Structure

### Single-Observable Design (vs. multi-observable V2-V4)

Focus on ONE α at a time with MULTIPLE measurements:

```julia
for α in α_vec
    N, P = compute_N_P(α)
    
    # Initialize patterns once
    patterns = randn(N, P, N_TRIALS)
    target = patterns[:, 1, :]
    
    for T in T_vec  # Dense T sampling near T_c
        # Protocol: Start near target, equilibrate heavily
        state = target + 0.05 * noise
        
        # Equilibrate at this T
        for eq_step in 1:N_EQ_ref(T)
            mc_step!(state, ...)
        end
        
        # Sample observables
        measure_free_energy(state, T)
        measure_escape_time(state, T)
        measure_order_statistics(state, T)
        measure_acceptance(state, T)
        
        # Stream results to CSV immediately
        write_row(csv, α, T, results)
    end
end
```

### Why This Design Overcomes V1-V4 Issues

- **No blue bay**: Heating within each α (state initialized once, warmed)
- **No independent failures**: Each T builds on equilibrated state from T-δT
- **Efficient**: Focus CPU on critical α values where phase transition occurs
- **Observables clarify mechanism**: Not just "is it retrieval?" but "why does it fail?"

---

## Parameters for Paper

**Grid for boundary characterization:**
- α: [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]  (7 values, focus on physics)
- T: Log-spaced 50 points from 0.05 → 2.5, WITH DENSE SAMPLING near predicted T_c(α)
- N_TRIALS: 256 (good statistics, LSR-scale)
- N_EQ_ref(T): Heating protocol from coldest T
- N_SAMP: 500-1000 per T (boundary characterization, not full phase diagram)

**Expected output:**
- `lsr_boundary_v5.csv`: α, T, E_mean, E_std, φ_mean, φ_std, τ_esc, accept_rate
- `lse_boundary_v5.csv`: Same structure (for comparison)
- `phase_boundary_overlay.pdf`: E(φ) vs T for different α (visualization of minimum collapse)

---

## Computational Cost (24-48h node)

**Per α**: ~2-3 hours (50 T values × 256 trials, heavy sampling)
**Total**: 7 α × 2.5 h = 17.5 h → leaves margin for LSE validation

**Alternative if needed**: 
- Reduce to 4 α [0.02, 0.15, 0.35, 0.50] → 10-12h total
- Or run v3 phase diagrams in parallel on another node

---

## Why This Answers "What Destroys the Local Minimum?"

| Measurement | What It Reveals |
|---|---|
| E(T) curves | Thermodynamic instability point (dF/dφ = 0 moves) |
| τ_esc(T) | Kinetic barrier collapse (explicit time scale) |
| φ statistics | Order parameter vanishes at T_c (smoking gun) |
| a(T) jump | Accessibility changes (more states reachable) |

Combined: **Direct characterization of phase transition** underlying "blue bay" in V2/V1.
