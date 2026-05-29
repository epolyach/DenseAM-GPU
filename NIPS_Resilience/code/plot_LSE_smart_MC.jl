#=
Smart LSE Monte Carlo at α=1 with a FIXED storage budget
────────────────────────────────────────────────────────────────────────
Naive MC at α=1 stores M = exp(αN) = exp(N) patterns.
At N=12 this is ≈ 163 000 patterns — the budget that fit in our prior runs.

Question (reframed): if we KEEP THIS BUDGET K_budget = exp(12) constant,
but increase N, how does the picture change?

For α=1, the full population has M = exp(N) i.i.d. patterns with
  φ_μ ~ f(φ) = C_N(1-φ²)^((N-3)/2)   ⇔   (1+φ)/2 ~ Beta((N-1)/2,(N-1)/2).
Keeping only the K_budget patterns with the largest φ_μ amounts to a cut
  φ > φ_cut(N),       with   P(φ > φ_cut) = K_budget / M = exp(12 - N).

We compute:
  1.  φ_cut(N)                                   – minimum retained overlap
  2.  φ_max(N) and δφ_cut(N) = φ_max - φ_cut     – retained window width
  3.  fraction of  p(x) = Σ exp(-N(1-φ_μ))  that the truncation discards
  4.  exact density f(φ) vs the Gaussian approximation  N(0,1/N)
      (also weighted by the LSE kernel exp(-N(1-φ))).

Outputs
  ../panels_paper/smart_MC_budget_curves.{png,pdf}
  ../panels_paper/smart_MC_density_vs_gauss.{png,pdf}
  – plus a text table to stdout.
────────────────────────────────────────────────────────────────────────
=#

ENV["GKSwstype"] = "100"   # headless GR backend

using Plots
using SpecialFunctions       # logbeta, beta_inc, beta_inc_inv
using Printf
using QuadGK

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

const FIG_DPI    = 300
const FIG_W      = round(Int, 86 / 25.4 * 100)
const FIG_H      = round(Int, FIG_W * 0.75)
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_LEG   = 6

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK, legendfontsize=FONT_LEG)

# ─────────────── Problem setup ───────────────
const α          = 1.0
const N_BUDGET   = 12                       # the N we actually ran in prior MC
const K_BUDGET   = exp(N_BUDGET)            # ≈ 162 754 patterns
const Ns         = [12, 15, 18, 22, 28, 35, 45, 60, 80, 100, 140, 200]

# φ_max with finite-N correction
phi_max(α, N) = sqrt(max(0.0, 1 - exp(-2 * α / (1 - 3/N))))

# Beta((N-1)/2, (N-1)/2) helpers via SpecialFunctions ───────────────
beta_a(N) = (N - 1) / 2
# log pdf of Beta(a,a) at x ∈ (0,1)
log_beta_pdf(x, a) = (a - 1) * (log(x) + log1p(-x)) - logbeta(a, a)
beta_pdf(x, a) = exp(log_beta_pdf(x, a))

# Log of upper tail P(X > x) for Beta(a,a)
# Falls back to a log-space integral when the direct call underflows
# (needed for deep tails, where  P  may be e^{-100} or smaller).
function log_beta_sf(x, a)
    x <= 0 && return 0.0
    x >= 1 && return -Inf
    # try direct
    if x < 0.95
        _, q = beta_inc(a, a, x)
        q > 0 && return log(q)
    end
    # deep upper tail: u = 1-x small.  Write
    #   P(X>x) = u^a / B(a,a) · ∫_0^1 t^{a-1}(1-u t)^{a-1} dt
    u = 1 - x
    integrand(t) = t > 0 ? exp((a-1)*(log(t) + log1p(-u*t))) : 0.0
    val, _ = quadgk(integrand, 0.0, 1.0; rtol=1e-10)
    return a*log(u) + log(val) - logbeta(a, a)
end

# x such that log P(X > x) = log_p, with Newton refinement in log-u space
function beta_isf_log(log_p, a)
    log_p >= 0  && return 0.0
    isinf(log_p) && log_p < 0 && return 1.0
    # initial guess from leading asymptotic:  a·log(u) ≈ log_p + log a + logbeta
    log_u = (log_p + log(a) + logbeta(a, a)) / a
    log_u = min(log_u, log(0.5))            # u < 0.5 means x > 0.5
    for _ in 1:60
        u = exp(log_u)
        u = clamp(u, 1e-300, 0.5)
        x = 1 - u
        lp = log_beta_sf(x, a)
        err = lp - log_p
        abs(err) < 1e-10 && break
        # d log P / d log u  = u · pdf(x) / P   (positive)
        log_pdf = (a-1) * (log(x) + log(u)) - logbeta(a, a)
        dlogP_dlogu = exp(log_pdf + log(u) - lp)
        dlogP_dlogu < 1e-12 && (dlogP_dlogu = a)   # safety
        log_u -= err / dlogP_dlogu
    end
    return 1 - exp(log_u)
end

# φ_cut such that P(φ > φ_cut) = K_budget / M ; for N ≤ N_BUDGET keep all (φ_cut = -1)
function phi_cut(N; α=α, K=K_BUDGET)
    log_p_keep = log(K) - α * N
    log_p_keep >= 0 && return -1.0
    x_cut = beta_isf_log(log_p_keep, beta_a(N))
    return 2*x_cut - 1
end

# Per-pattern weight integral   I(φ_lo) = E_f[ exp(-N(1-φ)) · 1{φ > φ_lo} ]
# In Beta(x) coordinates with x = (1+φ)/2,   1-φ = 2(1-x):
#   I = ∫_{x_lo}^{1} exp(-2N(1-x)) · pdf_Beta(x) dx
# Computed in log-space then summed via quadrature for stability.
function I_weight_above(φ_lo, N)
    a   = beta_a(N)
    x_lo = clamp((1 + φ_lo)/2, 0.0, 1.0)
    f(x) = exp(-2*N*(1 - x) + log_beta_pdf(x, a))
    val, _ = quadgk(f, x_lo, 1.0; rtol=1e-10)
    return val
end

# ─────────────── Build the table ───────────────
println()
println("α = $α    K_budget = exp($N_BUDGET) ≈ $(round(K_BUDGET, digits=0))")
println()
@printf("  %3s  %12s  %8s  %8s  %8s  %12s  %10s  %10s\n",
        "N", "M=e^N", "φ_max", "φ_cut", "δφ_cut", "e^{-N·δφ_cut}",
        "I_below/I", "I_above/I")
println("  " * "─"^88)

φcuts   = Float64[]
δφcuts  = Float64[]
fdisc   = Float64[]
fkeep   = Float64[]
M_arr   = Float64[]

for N in Ns
    φM   = phi_max(α, N)
    φc   = phi_cut(N)
    δφc  = φM - φc
    M    = exp(α * N)
    rel_w_cut = exp(-N * max(δφc, 0.0))     # weight of marginal kept pattern
                                            # relative to the champion

    I_full  = I_weight_above(-1.0, N)
    I_above = I_weight_above(φc,    N)
    I_below = max(I_full - I_above, 0.0)
    frac_disc = I_below / I_full
    frac_keep = I_above / I_full

    push!(φcuts, φc); push!(δφcuts, δφc); push!(fdisc, frac_disc)
    push!(fkeep, frac_keep); push!(M_arr, M)

    @printf("  %3d  %12.3e  %8.4f  %8.4f  %8.4f  %12.3e  %10.3e  %10.3e\n",
            N, M, φM, φc, δφc, rel_w_cut, frac_disc, frac_keep)
end

println()
println("Reading: at α=1, smart-MC with the same budget as the N=12 run")
println("can reach much larger N, but the *discarded* fraction of the")
println("LSE sum p(x) tells whether we are still resolving capacity faithfully.")

# ─────────────── Plot 1: φ_cut, δφ_cut, discarded weight vs N ───────────────
println("\nPlotting budget curves...")
φmaxs = [phi_max(α, N) for N in Ns]

p1 = plot(layout=(1,3),
          size=(FIG_W*3 + 60, FIG_H), dpi=FIG_DPI,
          left_margin=4Plots.mm, bottom_margin=4Plots.mm, top_margin=2Plots.mm)

# (a) φ_cut and φ_max
plot!(p1[1], Ns, φmaxs, lw=1.8, color=:black, marker=:circle, ms=3,
      label="φ_max(N)")
plot!(p1[1], Ns, φcuts, lw=1.8, color=:red,   marker=:diamond, ms=3,
      label="φ_cut(N), budget=e^$N_BUDGET")
hline!(p1[1], [-1, 1], color=:gray, ls=:dot, lw=0.5, label=false)
plot!(p1[1], xlabel="N", ylabel="φ",
      title="(a)  retained window at α=1", titlefontsize=FONT_GUIDE,
      legend=:bottomright, ylims=(-1.05, 1.05))

# (b) δφ_cut = φ_max - φ_cut
plot!(p1[2], Ns, δφcuts, lw=1.8, color=:purple, marker=:square, ms=3,
      label="δφ_cut = φ_max - φ_cut")
plot!(p1[2], xlabel="N", ylabel="δφ_cut",
      title="(b)  window width", titlefontsize=FONT_GUIDE,
      legend=:topright)

# (c) discarded weight fraction
plot!(p1[3], Ns, fdisc, lw=1.8, color=:crimson, marker=:utriangle, ms=3,
      label="I_discarded / I_total")
plot!(p1[3], xlabel="N", ylabel="discarded weight fraction",
      title="(c)  weight thrown from p(x)", titlefontsize=FONT_GUIDE,
      legend=:topleft, yscale=:log10,
      ylims=(max(1e-30, minimum(filter(x->x>0, fdisc)) / 10), 1.5))

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir, "smart_MC_budget_curves.$ext"))
end
println("Saved: panels_paper/smart_MC_budget_curves.{png,pdf}")

# ─────────────── Plot 2: density f(φ) vs Gaussian, raw and weighted ───────────────
println("Plotting density vs Gaussian comparison...")

# Gaussian approximation:  f_G(φ) = sqrt(N/(2π)) · exp(-N φ²/2)
# Exact:                   f_E(φ) = C_N (1-φ²)^((N-3)/2)
# Weighted: multiply by exp(-N(1-φ)) (the LSE kernel at α=1, b=N)

function f_exact(φ, N)
    x = (1 + φ)/2
    (x <= 0 || x >= 1) && return 0.0
    return beta_pdf(x, beta_a(N)) / 2     # Jacobian dφ = 2 dx
end

f_gauss(φ, N) = sqrt(N/(2π)) * exp(-N * φ^2 / 2)

p2 = plot(layout=(2,3), size=(FIG_W*3 + 60, FIG_H*2 + 20), dpi=FIG_DPI,
          left_margin=4Plots.mm, bottom_margin=4Plots.mm, top_margin=2Plots.mm)

Ns_panel = [25, 50, 100]
for (j, N) in enumerate(Ns_panel)
    # Top row: raw density (log scale to expose the tail Gaussian misses)
    φs = range(-0.999, 0.999, length=601)
    fE = [f_exact(φ, N) for φ in φs]
    fG = [f_gauss(φ, N) for φ in φs]
    φM = phi_max(α, N)
    φc = phi_cut(N)

    plot!(p2[1, j], φs, fE, lw=1.8, color=:black, label="exact f(φ)")
    plot!(p2[1, j], φs, fG, lw=1.5, color=:blue,  ls=:dash, label="Gaussian N(0,1/N)")
    vline!(p2[1, j], [φM], color=:darkgreen, lw=1.2, ls=:dot, label="φ_max")
    vline!(p2[1, j], [φc], color=:red,       lw=1.2, ls=:dot, label="φ_cut")
    plot!(p2[1, j], xlabel="φ", ylabel="density",
          title=@sprintf("density, N=%d  α=1", N), titlefontsize=FONT_GUIDE,
          yscale=:log10, ylims=(1e-20, 50),
          xlims=(-1.02, 1.02), legend=(j==1 ? :topleft : false))

    # Bottom row: LSE-weighted density  w(φ) = exp(-N(1-φ)) · f(φ)
    # Normalize by I_full for the EXACT distribution so the curves are comparable
    I_full_E = I_weight_above(-1.0, N)
    wE = [exp(-N*(1-φ)) * fE[i] / I_full_E for (i,φ) in enumerate(φs)]
    # Gaussian-weighted (with its own normalization):
    #   ∫ exp(-N(1-φ)) sqrt(N/(2π)) exp(-Nφ²/2) dφ over (-∞,∞)
    #   = exp(-N) · ∫ sqrt(N/(2π)) exp(-N(φ²-2φ)/2) dφ
    #   = exp(-N) · exp(N/2) = exp(-N/2)
    I_full_G = exp(-N/2)
    wG = [exp(-N*(1-φ)) * fG[i] / I_full_G for (i,φ) in enumerate(φs)]

    plot!(p2[2, j], φs, wE, lw=1.8, color=:black,  label="exact, normalized")
    plot!(p2[2, j], φs, wG, lw=1.5, color=:blue, ls=:dash,
          label="Gaussian, normalized")
    vline!(p2[2, j], [φc], color=:red, lw=1.2, ls=:dot, label="φ_cut")
    plot!(p2[2, j], xlabel="φ",
          ylabel="weighted density (norm.)",
          title=@sprintf("LSE-weighted contribution, N=%d", N),
          titlefontsize=FONT_GUIDE,
          xlims=(-0.05, 1.02), legend=(j==1 ? :topleft : false))
end

for ext in ("png", "pdf")
    savefig(p2, joinpath(out_dir, "smart_MC_density_vs_gauss.$ext"))
end
println("Saved: panels_paper/smart_MC_density_vs_gauss.{png,pdf}")

println("\nDone.")
