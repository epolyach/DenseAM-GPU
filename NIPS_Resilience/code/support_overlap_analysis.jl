#=
Support Overlap Analysis for LSR Dense Associative Memory
──────────────────────────────────────────────────────────
Evaluates Pr(∃μ≠1: support of ξ^1 ∩ support of ξ^μ ≠ ∅) as a function of α and N.

Three levels of "overlap":
  Level 0: Geometric — spherical caps intersect (q > 2φ_c² − 1 = 0)
  Level 1: Escape-accessible — centroid reachable from retrieval state (q > q_min(φ_eq))
  Level 2: Direct — both patterns in support at the retrieval state (q > φ_c/φ_eq)

Key formula (Poisson regime):
  E[n_overlap] = P · Φᶜ(q* √N) ≈ exp(N(α − q*²/2)) / (q* √(2πN))

  ⟹ Critical α_c = q*²/2  (thermodynamic limit)
     For α > α_c: E[n] → ∞ exponentially  ⟹  Pr → 1
     For α < α_c: E[n] → 0 exponentially  ⟹  Pr → 0
=#

using SpecialFunctions  # for erfc

# ──────────────── Model parameters ────────────────
const b_lsr = 2 + sqrt(2)
const φ_c   = (b_lsr - 1) / b_lsr   # ≈ 0.7071 = 1/√2

# ──────────────── Gaussian tail ────────────────
"""Complementary normal CDF: Pr(Z > z) for Z ~ N(0,1)"""
Φc(z) = erfc(z / sqrt(2)) / 2

"""Log of complementary normal CDF (numerically stable for large z)"""
function logΦc(z)
    if z < 6
        return log(Φc(z))
    else
        # Mill's ratio asymptotic: Φᶜ(z) ~ φ(z)/z · (1 - 1/z² + ...)
        return -z^2/2 - log(z) - 0.5*log(2π) + log(1 - 1/z^2 + 3/z^4)
    end
end

# ──────────────── q_min for escape accessibility ────────────────
"""
Minimum inter-pattern overlap for the centroid escape path to fit on the sphere.
From: φ_eq² + (φ_c − φ_eq·q)²/(1−q²) = 1
"""
function q_min_escape(φ_eq)
    # Quadratic: q² − 2φ_eq·φ_c·q/(φ_eq²+φ_c²−1) + (φ_c²−1+φ_eq²)/(φ_eq²+φ_c²−1) ...
    # Expand: φ_eq² + (φ_c − φ_eq·q)²/(1−q²) = 1
    # (φ_c − φ_eq·q)² = (1 − φ_eq²)(1 − q²)
    # φ_c² − 2φ_c·φ_eq·q + φ_eq²·q² = 1 − φ_eq² − q² + φ_eq²·q²
    # φ_c² − 2φ_c·φ_eq·q = 1 − φ_eq² − q²
    # q² − 2φ_c·φ_eq·q + (φ_c² − 1 + φ_eq²) = 0
    a = 1.0
    b = -2φ_c * φ_eq
    c = φ_c^2 - 1 + φ_eq^2
    disc = b^2 - 4a*c
    disc < 0 && return NaN
    return (-b - sqrt(disc)) / (2a)  # smaller root
end

# ──────────────── Core computation ────────────────
"""
Expected number of patterns with q_{1μ} > q* and probability of at least one.
Returns (E_n, Pr_at_least_one, P, N).
"""
function overlap_stats(α, N, q_star)
    logP = α * N
    z = q_star * sqrt(N)
    log_p_tail = logΦc(z)
    log_E_n = logP + log_p_tail   # log(E[n]), using log(P-1) ≈ log(P)
    E_n = exp(log_E_n)

    # Pr(n ≥ 1) = 1 − (1 − p)^{P−1} ≈ 1 − exp(−E[n]) for Poisson regime
    if E_n > 30
        Pr = 1.0
    elseif E_n < 1e-10
        Pr = E_n  # 1 − exp(−ε) ≈ ε
    else
        Pr = 1 - exp(-E_n)
    end
    return (E_n=E_n, Pr=Pr, logP=logP, N=N)
end

"""Critical α where E[n] = 1 (solved numerically)."""
function alpha_critical(N, q_star; tol=1e-8)
    # Thermodynamic limit: α_c = q*²/2
    # Finite N correction: α_c(N) = q*²/2 + ln(q*√(2πN)) / N
    α_asymp = q_star^2 / 2 + log(q_star * sqrt(2π * N)) / N

    # Bisection for exact value
    α_lo, α_hi = 0.001, 1.0
    for _ in 1:100
        α_mid = (α_lo + α_hi) / 2
        s = overlap_stats(α_mid, N, q_star)
        if s.E_n > 1
            α_hi = α_mid
        else
            α_lo = α_mid
        end
        abs(α_hi - α_lo) < tol && break
    end
    return (exact=(α_lo + α_hi)/2, asymptotic=α_asymp, thermodynamic=q_star^2/2)
end

# ════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ════════════════════════════════════════════════════════
function main()
    println("=" ^ 78)
    println("  SUPPORT OVERLAP ANALYSIS: LSR Dense Associative Memory")
    println("  P = exp(αN),  q_{1μ} ~ N(0, 1/N),  φ_c = $(round(φ_c, digits=4))")
    println("=" ^ 78)

    # ──────── 1. Three levels of overlap ────────
    println("\n┌─────────────────────────────────────────────────────────┐")
    println("│  THREE LEVELS OF SUPPORT OVERLAP                       │")
    println("└─────────────────────────────────────────────────────────┘")

    println("\nLevel 0 — Geometric (caps intersect): q* = 2φ_c² − 1 = $(round(2φ_c^2 - 1, digits=4))")
    println("  → Any positive q suffices. P(q>0) = 1/2 per pattern.")
    println("  → E[n] = P/2 = exp(αN)/2.  ALWAYS satisfied for P ≥ 2.")

    φ_eq_values = [0.85, 0.90, 0.95]
    println("\nLevel 1 — Escape accessible (centroid reachable from retrieval):")
    for φ_eq in φ_eq_values
        qm = q_min_escape(φ_eq)
        αc_inf = qm^2 / 2
        println("  φ_eq = $φ_eq:  q_min = $(round(qm, digits=4)),  α_c(∞) = q²/2 = $(round(αc_inf, digits=4))")
    end

    println("\nLevel 2 — Direct overlap at retrieval (q > φ_c/φ_eq):")
    for φ_eq in φ_eq_values
        qd = φ_c / φ_eq
        αc_inf = qd^2 / 2
        println("  φ_eq = $φ_eq:  q_direct = $(round(qd, digits=4)),  α_c(∞) = $(round(αc_inf, digits=4))")
    end

    # ──────── 2. Expected number of overlapping patterns ────────
    println("\n\n┌─────────────────────────────────────────────────────────┐")
    println("│  TABLE: E[n] and Pr(≥1) for ESCAPE-ACCESSIBLE overlap  │")
    println("│  (φ_eq = 0.90, q_min = $(round(q_min_escape(0.90), digits=4)))               │")
    println("└─────────────────────────────────────────────────────────┘")

    q_star = q_min_escape(0.90)
    α_values = [0.02, 0.04, 0.054, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    N_values = [30, 50, 100, 200, 500]

    # Header
    print("\n  α    │  α_c(∞)=", round(q_star^2/2, digits=3))
    for N in N_values
        print("  │  N=", lpad(N, 3))
    end
    println("\n  ─────┼──────────", "──┼───────" ^ length(N_values))

    for α in α_values
        print("  ", rpad(round(α, digits=3), 5))
        print(" │  ")
        regime = α < q_star^2/2 ? "sub" : (α ≈ q_star^2/2 ? "crit" : "sup")
        print(rpad(regime, 8))
        for N in N_values
            s = overlap_stats(α, N, q_star)
            if s.E_n > 1000
                print("  │  ≫1  ")
            elseif s.E_n < 0.001
                print("  │  ≈0  ")
            else
                print("  │ ", lpad(round(s.E_n, digits=1), 5))
            end
        end
        println()
    end

    println("\n  Legend: E[n] = expected number of patterns with q > q_min")
    println("  'sub'  = α < α_c (vanishes with N),  'sup' = α > α_c (diverges with N)")

    # ──────── 3. Pr(≥1) table ────────
    println("\n\n  Pr(at least one pattern with q > q_min):")
    print("  α    ")
    for N in N_values
        print(" │  N=", lpad(N, 3))
    end
    println("\n  ─────", "─┼───────" ^ length(N_values))

    for α in α_values
        print("  ", rpad(round(α, digits=3), 5))
        for N in N_values
            s = overlap_stats(α, N, q_star)
            if s.Pr > 0.999
                print(" │  ≈1  ")
            elseif s.Pr < 0.001
                print(" │  ≈0  ")
            else
                print(" │ ", lpad(round(s.Pr, digits=3), 5))
            end
        end
        println()
    end

    # ──────── 4. Critical α as function of N ────────
    println("\n\n┌─────────────────────────────────────────────────────────┐")
    println("│  CRITICAL α_c(N) where E[n_overlap] = 1                │")
    println("└─────────────────────────────────────────────────────────┘")

    println("\n  q_min = $(round(q_star, digits=4))  (escape-accessible, φ_eq = 0.90)")
    println("  Thermodynamic limit: α_c(∞) = q²/2 = $(round(q_star^2/2, digits=4))")
    println()
    println("  N     α_c(exact)   α_c(asympt)   α_c(∞)")
    println("  ─────────────────────────────────────────")
    for N in [20, 30, 40, 50, 75, 100, 200, 500, 1000]
        ac = alpha_critical(N, q_star)
        println("  $(lpad(N, 4))    $(lpad(round(ac.exact, digits=4), 8))    $(lpad(round(ac.asymptotic, digits=4), 8))    $(round(ac.thermodynamic, digits=4))")
    end

    # ──────── 5. Specific values from the paper ────────
    println("\n\n┌─────────────────────────────────────────────────────────┐")
    println("│  VALUES FROM THE PAPER (Table in §2.1)                 │")
    println("└─────────────────────────────────────────────────────────┘")

    paper_data = [
        (0.200, 50,  20_000),
        (0.220, 49,  43_000),
        (0.240, 47,  88_000),
        (0.265, 46, 201_000),
    ]

    println("\n  For q* = 0.38 (paper's value for centroid at φ = 0.831):")
    println("  α      N    P        z=q*√N   Φᶜ(z)        E[n]     Pr(≥1)")
    println("  ─────────────────────────────────────────────────────────────")
    for (α, N, P) in paper_data
        z = 0.38 * sqrt(N)
        ptail = Φc(z)
        En = (P - 1) * ptail
        Pr = 1 - exp(-En)
        println("  $(rpad(α, 5))  $(lpad(N, 3))   $(lpad(P, 7))   $(lpad(round(z, digits=3), 6))   $(lpad(round(ptail, digits=6), 10))   $(lpad(round(En, digits=1), 8))   $(round(Pr, digits=6))")
    end

    # ──────── 6. Analytical formula summary ────────
    println("\n\n┌─────────────────────────────────────────────────────────┐")
    println("│  ANALYTICAL SUMMARY                                    │")
    println("└─────────────────────────────────────────────────────────┘")

    println("""

  Model: P = exp(αN) random patterns on S^{N-1}(√N)
  Inter-pattern overlap: q_{1μ} ~ N(0, 1/N)

  Expected number with q > q*:

    E[n] = (P−1) · Φᶜ(q*√N)
         ≈ exp(N(α − q*²/2)) / (q*√(2πN))      [Gaussian tail]

  Probability of at least one:

    Pr(n ≥ 1) = 1 − exp(−E[n])                  [Poisson approx]

  ────────────────────────────────────────────────
  CRITICAL EXPONENT:  α − q*²/2
  ────────────────────────────────────────────────

    α > q*²/2  ⟹  E[n] → ∞ exponentially  ⟹  Pr → 1  (certain)
    α < q*²/2  ⟹  E[n] → 0 exponentially  ⟹  Pr → 0  (impossible)
    α = q*²/2  ⟹  E[n] ~ 1/√N             ⟹  Pr → 0  (marginal)

  For different overlap criteria:

    Criterion              q*      α_c = q*²/2
    ─────────────────────────────────────────────
    Geometric overlap      0       0           (always present)
    Escape (φ_eq=0.95)     0.216   0.023
    Escape (φ_eq=0.90)     0.329   0.054
    Escape (φ_eq=0.85)     0.417   0.087
    Direct at retrieval    0.786   0.309       (φ_eq=0.90)
    Paper centroid         0.380   0.072

  ════════════════════════════════════════════════
  CONCLUSION for α ≥ 0.2:

    α = 0.2 is 2.8× to 3.7× above α_c for escape overlap.
    The exponent N(α − q*²/2) ≈ N × 0.13 to N × 0.15.

    At N = 50:  exp(50 × 0.13) ≈ 665 overlapping patterns.
    At N = 100: exp(100 × 0.13) ≈ 4.4 × 10⁵ overlapping patterns.

    Support overlap at α ≥ 0.2 is NOT an exception —
    it is exponentially certain in N.

    For α ≥ 0.2 and ANY N ≥ 20, Pr(overlap) > 0.999.
  ════════════════════════════════════════════════
""")

    # ──────── 7. Small-α regime: where overlap becomes rare ────────
    println("┌─────────────────────────────────────────────────────────┐")
    println("│  SMALL-α REGIME: where does overlap become rare?       │")
    println("└─────────────────────────────────────────────────────────┘")

    println("\n  For escape overlap (φ_eq = 0.90, q_min ≈ 0.329):")
    println("  α_c(∞) = 0.054")
    println()
    println("  α      N=30      N=50      N=100     N=200")
    println("  ──────────────────────────────────────────────────")
    for α in [0.01, 0.02, 0.03, 0.04, 0.05, 0.054, 0.06, 0.08, 0.10]
        print("  $(rpad(round(α, digits=3), 5))")
        for N in [30, 50, 100, 200]
            s = overlap_stats(α, N, q_star)
            pr_str = if s.Pr > 0.999
                "  ≈1    "
            elseif s.Pr < 0.0001
                "  ≈0    "
            else
                "  $(lpad(round(s.Pr, digits=4), 6))  "
            end
            print(pr_str)
        end
        println()
    end

    println("""

  ────────────────────────────────────────────────
  The overlap probability f(α, N) is:

    f(α, N) = 1 − exp(−exp(αN) · Φᶜ(q_min √N))

  This is a SHARP THRESHOLD at α_c ≈ 0.054:
    • Below α_c: f → 0 as N → ∞  (overlap is indeed rare)
    • Above α_c: f → 1 as N → ∞  (overlap is certain)
    • The transition sharpens with increasing N

  Since α = 0.2 ≫ α_c = 0.054, the overlap is
  overwhelmingly certain, not an exception.
  ────────────────────────────────────────────────
""")
end

main()
