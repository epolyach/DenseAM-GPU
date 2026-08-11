#=
ΔF/T vs T for several α values, with explanation of non-monotonicity.
Also: τ_eff vs T, and three-regime analysis.
────────────────────────────────────────────────────────────────────────
Output: panels_paper/barrier_vs_T.{png,pdf}
        panels_paper/tau_vs_T.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf
using LaTeXStrings

const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)
const FIG_H = FIG_W
const FONT_GUIDE = 8
const FONT_TICK = 7
const FONT_ANN = 7
const FONT_LEG = 6

default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK)

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

const b_lsr = 2 + sqrt(2)
const φ_c = (b_lsr - 1) / b_lsr

function φ_eq_LSR(T)
    T < 1e-10 && return 1.0
    φ = 0.95
    for _ in 1:200
        D = 1 - b_lsr + b_lsr*φ
        D ≤ 1e-10 && (φ = φ_c + 0.005; continue)
        f = (1 - φ^2) - T*φ*D
        fp = -2φ - T*(D + b_lsr*φ)
        φ = clamp(φ - f/fp, φ_c + 1e-8, 1 - 1e-8)
    end
    return φ
end

φ_1max(α) = sqrt(1 - exp(-2α))

function barrier_dFT(N, φeq, φ1m)
    v = (φ_c - φeq*φ1m)/sqrt(1-φ1m^2)
    v ≤ 0 && return 0.0
    R2 = 1 - φeq^2
    (R2 ≤ 0 || v^2 ≥ R2) && return Inf
    return (N-3)/2 * (-log(1 - v^2/R2))
end

τ_rel(N, φeq, T) = (1 - φeq^2) * N^2 / (2.88 * T^2)

# ──────────────── Panel 1: ΔF/T vs T ────────────────
T_range = range(0.05, 2.0, length=200)

alphas_plot = [0.20, 0.24, 0.28]
colors = [:crimson, :forestgreen, :darkorange]
M = 20000

p1 = plot(xlabel=L"T", ylabel=L"\Delta F / T",
    legend=:topright, legendfontsize=FONT_LEG,
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=1Plots.mm)

for (i, α) in enumerate(alphas_plot)
    N = floor(Int, log(M)/α)
    φ1m = φ_1max(α)
    dFT = [barrier_dFT(N, φ_eq_LSR(T), φ1m) for T in T_range]

    # T* where barrier is minimum: φ_eq(T*) = φ_{1μ}/φ_c
    T_star_φeq = φ1m / φ_c
    # find T* by scanning
    T_star = NaN
    for T in T_range
        if φ_eq_LSR(T) ≤ T_star_φeq
            T_star = T; break
        end
    end

    plot!(p1, T_range, dFT, color=colors[i], lw=2,
        label=@sprintf("α=%.2f (N=%d)", α, N))

    # Mark T*
    if !isnan(T_star)
        dF_star = barrier_dFT(N, φ_eq_LSR(T_star), φ1m)
        scatter!(p1, [T_star], [dF_star], color=colors[i],
            markersize=6, markershape=:star5, label=false)
    end
end

# Annotate the formula
annotate!(p1, 1.5, 12.0,
    text(L"\phi_\mathrm{eq}(T^*) = \varphi_{1\mu}/\varphi_c", FONT_ANN, :center, :gray30))

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir, "barrier_vs_T.$ext"))
end
println("Saved barrier_vs_T")

# ──────────────── Panel 2: τ_eff vs T ────────────────
const C_BARRIER = 0.37
const T_MC_v8m = 2^15 + 2^13

p2 = plot(xlabel=L"T", ylabel=L"\ln(\tau_\mathrm{eff})",
    legend=:topright, legendfontsize=FONT_LEG,
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=1Plots.mm)

for (i, α) in enumerate(alphas_plot)
    N = floor(Int, log(M)/α)
    φ1m = φ_1max(α)
    ln_τ = Float64[]
    for T in T_range
        φeq = φ_eq_LSR(T)
        dF = barrier_dFT(N, φeq, φ1m)
        tr = τ_rel(N, φeq, T)
        τ_eff = N * tr * exp(C_BARRIER * dF)
        push!(ln_τ, log(τ_eff))
    end
    plot!(p2, T_range, ln_τ, color=colors[i], lw=2,
        label=@sprintf("α=%.2f (N=%d)", α, N))
end

# T_MC line
hline!(p2, [log(T_MC_v8m)], color=:gray, lw=1, ls=:dash,
    label=L"\ln T_\mathrm{MC}")

for ext in ("png", "pdf")
    savefig(p2, joinpath(out_dir, "tau_vs_T.$ext"))
end
println("Saved tau_vs_T")

# ──────────────── Print T* values ────────────────
println("\nBarrier minimum T* (where φ_eq = φ_{1μ}/φ_c):")
for α in alphas_plot
    φ1m = φ_1max(α)
    target = φ1m / φ_c
    N = floor(Int, log(M)/α)
    for T in range(0.01, 3.0, length=1000)
        if φ_eq_LSR(T) ≤ target
            dF = barrier_dFT(N, φ_eq_LSR(T), φ1m)
            @printf("  α=%.2f: T*=%.2f, φ_eq(T*)=%.4f = φ_{1μ}/φ_c=%.4f, ΔF/T_min=%.2f\n",
                α, T, φ_eq_LSR(T), target, dF)
            break
        end
    end
end

println("\nDone.")
