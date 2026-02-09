using CairoMakie

# ──────────────── Theoretical equilibrium alignment ────────────────

# Eq. 33: φ_LSE(T) = ½[-T + √(T² + 4)]
function phi_lse_eq(T::Real)
    return 0.5 * (-T + sqrt(T^2 + 4))
end

# Eq. 37: (bT+1)y² - (2+T+Tb)y + T = 0,  φ = 1 - y
function phi_lsr_eq(T::Real, b::Real)
    A = b*T + 1
    B = -(2 + T + T*b)
    C = T
    disc = B^2 - 4*A*C
    disc < 0 && return NaN
    y = (-B - sqrt(disc)) / (2*A)
    phi = 1 - y
    (phi > 1 || phi < 0) && return NaN
    return phi
end

# ──────────────── Compute curves ────────────────

b = 2 + sqrt(2)
phi_c = (b - 1) / b  # ≈ 0.707

T_vec = range(0.001, 2.0, length=500)
phi_lse = [phi_lse_eq(T) for T in T_vec]
phi_lsr = [phi_lsr_eq(T, b) for T in T_vec]

# ──────────────── Production settings (1-column figure) ────────────────

fontsize_label = 26
fontsize_tick = 22
fontsize_legend = 20

fig = Figure(size=(800, 550), fontsize=fontsize_tick, figure_padding=15)

ax = Axis(fig[1, 1],
          xlabel="Temperature T", ylabel="Equilibrium alignment φ",
          xlabelsize=fontsize_label, ylabelsize=fontsize_label,
          xticklabelsize=fontsize_tick, yticklabelsize=fontsize_tick,
          xticks=0.0:0.5:2.0,
          xminorticks=0.0:0.1:2.0, xminorticksvisible=true,
          yticks=0.0:0.2:1.0,
          yminorticks=0.0:0.05:1.0, yminorticksvisible=true,
          limits=((0, 2.0), (0.0, 1.02)))

# LSE (Eq. 33)
lines!(ax, collect(T_vec), phi_lse,
       color=:blue, linewidth=2.5,
       label="LSE")

# LSR (Eq. 37)
lines!(ax, collect(T_vec), phi_lsr,
       color=:red, linewidth=2.5,
       label="LSR (b = $(round(b, digits=2)))")

# Hard wall (LSR support boundary)
hlines!(ax, [phi_c],
        color=:gray40, linewidth=1.5, linestyle=:dot,
        label="φ_c = $(round(phi_c, digits=2))")

# Legend
axislegend(ax, position=:lb, labelsize=fontsize_legend, framevisible=true)

# ──────────────── Save ────────────────

save("phi_eq_vs_T_prod.png", fig, px_per_unit=2)
println("✓ PNG saved: phi_eq_vs_T_prod.png")

save("phi_eq_vs_T_prod.eps", fig)
println("✓ EPS saved: phi_eq_vs_T_prod.eps")
