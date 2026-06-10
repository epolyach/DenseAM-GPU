#=
Heatmap of <phi> − phi_eq(T) using only
basin_stab_LSE_semismart_AAAI_Nramp_M4.4e6.csv.

Per (α, T) cell:
  mean_phi = mean over disorders and both replicas (phi_a, phi_b)
  residual = mean_phi − phi_eq(T)

Output: panels_paper/heatmap_Nramp_M4.4e6.{png,pdf}
=#

using Printf
using Plots
using Statistics

const φ_star = (sqrt(5) - 1)/2
const g_max  = 0.5*log(φ_star) + φ_star          # ≈ 0.3774

phi_eq(T) = 0.5*(-T + sqrt(T^2 + 4))
fret(T)   = (1 - phi_eq(T)) - 0.5*T*log(1 - phi_eq(T)^2)

# Three analytical boundaries
ac_gauss(T) = 0.5 * (1 - fret(T))^2
ac_bd(T)    = -0.5 * log(1 - (1 - fret(T))^2)
ac_sd(T)    = 1 - g_max - fret(T)
# Spinodal
T_sp_LO(α)  = (α + g_max) >= 1 ? 0.0 : (1 - (α + g_max)^2) / (α + g_max)
ac_sp(T)    = phi_eq(T) - g_max

# ─────────── Read CSV ───────────
const CSV_IN = joinpath(@__DIR__, "basin_stab_LSE_semismart_AAAI_Nramp_M4.4e6.csv")

cells = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
Ns    = Dict{Tuple{Float64,Float64}, Int}()

open(CSV_IN, "r") do f
    for line in eachline(f)
        startswith(line, "#") && continue
        startswith(line, "alpha") && continue
        fs = split(line, ",")
        α = parse(Float64, fs[1])
        T = parse(Float64, fs[2])
        Nused = parse(Int, fs[3])
        φa = parse(Float64, fs[6])
        φb = parse(Float64, fs[7])
        key = (α, T)
        if !haskey(cells, key)
            cells[key] = Float64[]
            Ns[key] = Nused
        end
        push!(cells[key], φa, φb)
    end
end

αs = sort(unique(k[1] for k in keys(cells)))
Ts = sort(unique(k[2] for k in keys(cells)))
@printf("α grid: %d points from %.3f to %.3f\n", length(αs), first(αs), last(αs))
@printf("T grid: %d points from %.3f to %.3f\n", length(Ts), first(Ts), last(Ts))

# Build residual matrix Z[iT, iα] = <phi> − phi_eq(T)
Z = fill(NaN, length(Ts), length(αs))
for (iα, α) in enumerate(αs), (iT, T) in enumerate(Ts)
    haskey(cells, (α, T)) || continue
    isempty(cells[(α, T)]) && continue
    Z[iT, iα] = mean(cells[(α, T)]) - phi_eq(T)
end

# ─────────── Plot ───────────
p = heatmap(αs, Ts, Z,
            xlabel = "α",
            ylabel = "T",
            title  = "⟨φ⟩ − φ_eq(T), semismart Nramp (M_TARGET = 4.4e6)",
            c      = :RdBu,
            clims  = (-0.9, 0.9),
            colorbar_title = "⟨φ⟩ − φ_eq(T)",
            size   = (900, 540),
            framestyle = :box,
            titlefontsize = 11, guidefontsize = 11, tickfontsize = 9)

# Boundary curves (parameterised by T)
T_curve = collect(0.005:0.005:0.495)
plot!(p, [ac_gauss(t) for t in T_curve], T_curve,
      color = :blue, lw = 2.0, ls = :solid, label = "Gauss")
plot!(p, [ac_bd(t)    for t in T_curve], T_curve,
      color = :orange, lw = 2.0, ls = :dash, label = "exact")
plot!(p, [ac_sd(t)    for t in T_curve], T_curve,
      color = :red, lw = 2.0, ls = :solid, label = "saddle")

# Spinodal: α as function of T such that ac_sp(T) = α
plot!(p, [ac_sp(t) for t in T_curve], T_curve,
      color = :purple, lw = 1.8, ls = :dot, label = "spinodal")

xlims!(p, (0.20, 0.70))
ylims!(p, (0.0, 0.5))

# Save
outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
out_png = joinpath(outdir, "heatmap_Nramp_M4.4e6.png")
out_pdf = joinpath(outdir, "heatmap_Nramp_M4.4e6.pdf")
savefig(p, out_png); savefig(p, out_pdf)
println("Saved:")
println("  ", out_png)
println("  ", out_pdf)

# Print N range over the grid
N_min = minimum(values(Ns))
N_max = maximum(values(Ns))
@printf("N range used in this dataset: %d … %d\n", N_min, N_max)
