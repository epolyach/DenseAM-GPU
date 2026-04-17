#=
Plot LSE phase boundary: Gaussian vs Exact alignment distribution
────────────────────────────────────────────────────────────────────────
Output: panels_LSE/phase_boundary_comparison.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using Plots
using Printf

# ──────────────── Theory ────────────────

# Retrieval equilibrium overlap (ICML Eq. 33) — exact, no approximation
φ_LSE(T) = 0.5 * (-T + sqrt(T^2 + 4))

# Retrieval free energy density (ICML Eq. 34) — exact
f_ret(T) = let φ = φ_LSE(T); 1 - φ - (T/2)*log(1 - φ^2); end

# Phase boundary: Gaussian (ICML Eq. 35)
α_c_gauss(T) = 0.5 * (1 - f_ret(T))^2

# Phase boundary: Exact density
function α_c_exact(T)
    fr = f_ret(T)
    arg = fr * (2 - fr)  # = 1 - (1-fr)²
    arg ≤ 0 && return Inf
    return -0.5 * log(arg)
end

# ──────────────── Plot ────────────────

out_dir = "panels_LSE"
mkpath(out_dir)

T_range = range(0.01, 2.5, length=500)
αg = [α_c_gauss(T) for T in T_range]
αe = [min(α_c_exact(T), 1.0) for T in T_range]  # cap for plotting

p = plot(αg, T_range,
    color=:black, lw=2.5, label="Gaussian: α_c = ½(1−f_ret)²",
    xlabel="α = ln(M)/N", ylabel="T",
    xlims=(0, 0.65), ylims=(0, 2.5),
    legend=:topright, legendfontsize=8,
    size=(500, 400), dpi=300,
    left_margin=3Plots.mm, bottom_margin=3Plots.mm)

plot!(p, αe, T_range,
    color=:red, lw=2.5, ls=:dash,
    label="Exact: α_c = −½ln(1−(1−f_ret)²)")

# Mark α = 0.5
vline!(p, [0.5], color=:gray, lw=1, ls=:dot, label=false)
annotate!(p, 0.51, 0.15, text("α=0.5", 7, :left, :gray))

# Region labels
annotate!(p, 0.15, 1.5, text("Retrieval", 11, :center, :blue))
annotate!(p, 0.50, 1.5, text("Non-retrieval", 11, :center, :red))

# Title
title!(p, "LSE phase boundary: Gaussian vs Exact density")

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, "phase_boundary_comparison.$ext"))
end
println("Saved: $(out_dir)/phase_boundary_comparison.{png,pdf}")

# ──────────────── Print table ────────────────
println()
println("  T       φ_eq     f_ret    α_c^Gauss   α_c^exact   ratio")
println("  " * "─"^58)
for T in [0.001, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]
    fr = f_ret(T)
    ag = α_c_gauss(T)
    ae = α_c_exact(T)
    @printf("  %.3f   %.4f   %.4f    %.4f      %.4f     %.2f\n",
        T, φ_LSE(T), fr, ag, ae > 10 ? Inf : ae, ae/ag)
end
