#=
Energy profile along the geodesic from ξ^μ to ξ^ν
for the LSR energy with two patterns.

x-axis: ψ/θ ∈ [0,1]  (0 = ξ^μ, 1 = ξ^ν)
y-axis: E · b/N  (dimensionless)

Three regions along the geodesic:
  [0, (θ−θ_c)/θ]        only μ active
  [(θ−θ_c)/θ, θ_c/θ]    both active  ← corridor
  [θ_c/θ, 1]             only ν active
=#

using Plots, Printf

function energy_profile(ψ, θ, b)
    φ_c = (b - 1) / b
    φ_μ = cos(ψ)
    φ_ν = cos(θ - ψ)
    term_μ = max(b * (φ_μ - φ_c), 0.0)
    term_ν = max(b * (φ_ν - φ_c), 0.0)
    S = term_μ + term_ν
    S ≤ 0 && return NaN
    return -log(S)
end

# ── Parameters ──
b   = 2 + sqrt(2)
φ_c = (b - 1) / b
θ_c = acos(φ_c)

q_values = [0.05, 0.20, 0.50, φ_c]
q_labels = ["q = 0.05  (q ≈ q_geom = 0,  ΔE ≈ 2.14)",
            "q = 0.20",
            "q = 0.50  (wide corridor)",
            @sprintf("q = φ_c = %.3f  (ΔE = 0)", φ_c)]
colors = [:red, :royalblue, :darkgreen, :purple]

Emax = 2.5   # show all diamonds

p = plot(xlabel = "ψ / θ   (0 = ξᵘ,  1 = ξᵛ)",
         ylabel = "E · b/N",
         title  = @sprintf("Energy along geodesic,  b = %.2f,  φ_c = 1/√2,  q_geom = 0", b),
         legend = :topleft, legendfontsize = 9,
         ylims  = (-0.55, Emax),
         size   = (1000, 600), dpi = 150,
         framestyle = :box)

# ── Per-curve corridor shading ──
for (i, q) in enumerate(q_values)
    θ_q = acos(q)
    xL = (θ_q - θ_c) / θ_q
    xR = θ_c / θ_q
    if xL < xR
        plot!(p, Shape([xL, xR, xR, xL], [-0.55, -0.55, Emax, Emax]),
              fillcolor = colors[i], fillalpha = 0.06, linecolor = colors[i],
              linealpha = 0.15, linewidth = 0.5, label = false)
    end
end

# ── Energy curves ──
for (i, q) in enumerate(q_values)
    θ = acos(q)
    n = 800
    ψ_arr = range(0, θ, length = n)
    E_arr = [min(energy_profile(ψ, θ, b), Emax + 0.1) for ψ in ψ_arr]
    x = collect(ψ_arr) ./ θ

    plot!(p, x, E_arr, lw = 2.5, color = colors[i], label = q_labels[i])

    # corridor edges (diamonds = barrier)
    θ_edge_L = θ - θ_c
    θ_edge_R = θ_c
    for ψ_e in [θ_edge_L, θ_edge_R]
        if 0 < ψ_e < θ
            E_e = energy_profile(ψ_e, θ, b)
            scatter!(p, [ψ_e / θ], [min(E_e, Emax - 0.05)],
                     ms = 9, color = colors[i], markershape = :diamond,
                     markerstrokecolor = :black, markerstrokewidth = 1,
                     label = false)
        end
    end

    # midpoint (circle = col / saddle)
    E_mid = energy_profile(θ/2, θ, b)
    scatter!(p, [0.5], [E_mid],
             ms = 9, color = colors[i], markershape = :circle,
             markerstrokecolor = :black, markerstrokewidth = 1.5,
             label = false)
end

# ── Legend markers explanation ──
scatter!(p, [NaN], [NaN], ms = 9, color = :gray40, markershape = :diamond,
         markerstrokecolor = :black, markerstrokewidth = 1,
         label = "◆  corridor edge  (barrier ΔE)")
scatter!(p, [NaN], [NaN], ms = 9, color = :gray40, markershape = :circle,
         markerstrokecolor = :black, markerstrokewidth = 1.5,
         label = "●  midpoint  (col / ∇E = 0)")

# ── Annotations ──
# blue (q=0.20): label barrier and col
θ_20 = acos(0.20)
x_edge20 = (θ_20 - θ_c) / θ_20
E_edge20  = energy_profile(θ_20 - θ_c, θ_20, b)
annotate!(p, x_edge20 - 0.03, E_edge20 + 0.12,
          text("ΔE (barrier)", 8, :royalblue, :right))
E_mid20 = energy_profile(θ_20/2, θ_20, b)
annotate!(p, 0.50 + 0.03, E_mid20 - 0.09,
          text("col", 8, :royalblue, :left))
# red (q=0.05): label barrier value
θ_05 = acos(0.05)
E_edge05 = energy_profile(θ_05 - θ_c, θ_05, b)
annotate!(p, 0.50, min(E_edge05, Emax - 0.05) + 0.15,
          text(@sprintf("ΔE·b/N ≈ %.2f", E_edge05), 8, :red, :center))
# note at bottom: corridor = shaded region between diamonds
annotate!(p, 0.50, -0.44,
          text("shaded band = corridor (both patterns active) for each q", 7, :gray40, :center))

hline!(p, [0.0], color = :black, lw = 1, ls = :dot, label = false)

savefig(p, "barrier_profile.png")
println("✓ Saved: barrier_profile.png")
