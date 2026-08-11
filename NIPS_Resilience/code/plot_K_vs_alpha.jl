#=
Plot log₁₀(K) vs α for several temperatures
K = expected number of escape-enabling patterns
Output: panels_paper/K_vs_alpha.{png,pdf}
=#

using Plots, Printf, LaTeXStrings, SpecialFunctions
default(guidefontsize=8, tickfontsize=7, legendfontsize=7)

const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)
const FIG_H = FIG_W

const b_lsr = 2 + sqrt(2)
const φ_c = (b_lsr - 1) / b_lsr
const M = 20000

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

function φ_min(φeq)
    φ_c * φeq - sqrt((1 - φ_c^2) * (1 - φeq^2))
end

function compute_K(α, T)
    N = log(M) / α
    φeq = φ_eq_LSR(T)
    pm = φ_min(φeq)
    pm ≥ 1 && return 0.0
    pm = max(0.0, pm)
    logC = loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)
    n_pts = 2000
    dφ = (1.0 - pm) / n_pts
    dφ ≤ 0 && return 0.0
    integral = 0.0
    for i in 0:n_pts
        φ = pm + i * dφ
        φ ≥ 1 && break
        s = 1 - φ^2
        s ≤ 0 && continue
        integral += exp(logC + (N-3)/2 * log(s)) * dφ
    end
    return (M - 1) * integral
end

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

αs = range(0.02, 0.50, length=200)
Ts = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
colors = [:black, :purple, :blue, :teal, :green, :orange]
styles = [:dot, :solid, :solid, :solid, :solid, :solid]

# Match centroid_geometry.pdf: half-width, same aspect
p = plot(xlabel=L"\alpha", ylabel=L"\log_{10} K",
         xlims=(0, 0.50), ylims=(-2, 4),
         size=(FIG_W, round(Int, FIG_W * 0.85)), dpi=FIG_DPI,
         framestyle=:box, legend=:topleft, legendfontsize=6,
         left_margin=2Plots.mm, bottom_margin=2Plots.mm)

for (i, T) in enumerate(Ts)
    Ks = [compute_K(α, T) for α in αs]
    log10K = [K > 0 ? log10(K) : NaN for K in Ks]
    lbl = T == 0 ? "T=0" : (T < 0.1 ? @sprintf("T=%.2f", T) : @sprintf("T=%.1f", T))
    plot!(p, αs, log10K, lw=1.5, color=colors[i], ls=styles[i],
          label=lbl)
end

# K=1 line
hline!(p, [0], color=:gray, lw=1, ls=:dot, label=false)

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "K_vs_alpha.$ext"))
end
println("Saved: panels_paper/K_vs_alpha.{png,pdf}")
