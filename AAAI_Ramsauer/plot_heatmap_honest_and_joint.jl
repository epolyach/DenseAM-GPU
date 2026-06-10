#=
Two plots:
  1) Honest-only heatmap from basin_stab_LSE_honest_AAAI_N25.csv  (N=25 throughout)
  2) Joint heatmap: honest data overlays the semismart Nramp data
     (semismart fills the entire grid; honest values replace semismart
     in the cells where honest data exists, α ∈ [0.20, 0.56]).

Outputs:
  panels_paper/heatmap_honest_N25.{png,pdf}
  panels_paper/heatmap_honest_over_Nramp.{png,pdf}
=#

using Printf
using Plots
using Statistics

const φ_star = (sqrt(5) - 1)/2
const g_max  = 0.5*log(φ_star) + φ_star

phi_eq(T) = 0.5*(-T + sqrt(T^2 + 4))
fret(T)   = (1 - phi_eq(T)) - 0.5*T*log(1 - phi_eq(T)^2)
ac_gauss(T) = 0.5 * (1 - fret(T))^2
ac_bd(T)    = -0.5 * log(1 - (1 - fret(T))^2)
ac_sd(T)    = 1 - g_max - fret(T)
ac_sp(T)    = phi_eq(T) - g_max

# ─────────── Generic loader ───────────
function load_cells(path, idx_phi)
    cells = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
    open(path, "r") do f
        for line in eachline(f)
            startswith(line, "#") && continue
            startswith(line, "alpha") && continue
            fs = split(line, ",")
            length(fs) <= maximum(idx_phi) && continue
            α = parse(Float64, fs[1]); T = parse(Float64, fs[2])
            key = (α, T)
            haskey(cells, key) || (cells[key] = Float64[])
            for j in idx_phi
                push!(cells[key], parse(Float64, fs[j]))
            end
        end
    end
    return cells
end

# Build matrix indexed by (Ts, αs)
function build_matrix(cells, αs, Ts)
    Z = fill(NaN, length(Ts), length(αs))
    for (iα, α) in enumerate(αs), (iT, T) in enumerate(Ts)
        haskey(cells, (α, T)) || continue
        v = cells[(α, T)]
        isempty(v) && continue
        Z[iT, iα] = mean(v) - phi_eq(T)
    end
    return Z
end

# ─────────── Load both ───────────
const HONEST_CSV   = joinpath(@__DIR__, "basin_stab_LSE_honest_AAAI_N25.csv")
const SEMISMART_CSV = joinpath(@__DIR__, "basin_stab_LSE_semismart_AAAI_Nramp_M4.4e6.csv")

# honest columns: alpha,T,N_used,disorder,phi_a,phi_b,...  → φ in columns 5, 6
cells_honest   = load_cells(HONEST_CSV,   [5, 6])
# semismart columns: alpha,T,N_used,K_retained,disorder,phi_a,phi_b,... → 6, 7
cells_semi     = load_cells(SEMISMART_CSV, [6, 7])

αs_honest = sort(unique(k[1] for k in keys(cells_honest)))
αs_semi   = sort(unique(k[1] for k in keys(cells_semi)))
Ts_all    = sort(unique(vcat(
                  [k[2] for k in keys(cells_honest)],
                  [k[2] for k in keys(cells_semi)])))
αs_all    = sort(unique(vcat(αs_honest, αs_semi)))

@printf("Honest α range: %.3f … %.3f  (%d points)\n",
        first(αs_honest), last(αs_honest), length(αs_honest))
@printf("Semismart α range: %.3f … %.3f  (%d points)\n",
        first(αs_semi), last(αs_semi), length(αs_semi))

# Common T grid for both datasets
Z_honest = build_matrix(cells_honest, αs_all, Ts_all)
Z_semi   = build_matrix(cells_semi,   αs_all, Ts_all)

# Joint: honest where available, semismart elsewhere
Z_joint = copy(Z_semi)
for i in eachindex(Z_joint)
    if !isnan(Z_honest[i])
        Z_joint[i] = Z_honest[i]
    end
end

# ─────────── Plot helper ───────────
function make_plot(Z, αs, Ts, title)
    p = heatmap(αs, Ts, Z,
                xlabel = "α", ylabel = "T",
                title  = title,
                c      = :RdBu,
                clims  = (-0.9, 0.9),
                colorbar_title = "⟨φ⟩ − φ_eq(T)",
                size   = (900, 540),
                framestyle = :box,
                titlefontsize = 11, guidefontsize = 11, tickfontsize = 9)

    T_curve = collect(0.005:0.005:0.495)
    plot!(p, [ac_gauss(t) for t in T_curve], T_curve,
          color = :blue,   lw = 2.0, ls = :solid, label = "Gauss")
    plot!(p, [ac_bd(t)    for t in T_curve], T_curve,
          color = :orange, lw = 2.0, ls = :dash,  label = "exact")
    plot!(p, [ac_sd(t)    for t in T_curve], T_curve,
          color = :red,    lw = 2.0, ls = :solid, label = "saddle")
    plot!(p, [ac_sp(t)    for t in T_curve], T_curve,
          color = :purple, lw = 1.8, ls = :dot,   label = "spinodal")

    xlims!(p, (0.20, 0.70))
    ylims!(p, (0.0, 0.5))
    return p
end

# ─────────── Plot 1: honest only ───────────
p1 = make_plot(Z_honest, αs_all, Ts_all,
               "⟨φ⟩ − φ_eq(T), honest MC at N = 25  (α ≤ 0.56)")

# ─────────── Plot 2: honest overlays semismart ───────────
p2 = make_plot(Z_joint, αs_all, Ts_all,
               "⟨φ⟩ − φ_eq(T), honest N=25 overlay on semismart Nramp (M ≈ 4.4e6)")
# Mark the seam at α = max(αs_honest)
α_seam = maximum(αs_honest)
vline!(p2, [α_seam], lw = 1.2, ls = :dashdot, color = :black, alpha = 0.7,
       label = @sprintf("honest stops at α = %.2f", α_seam))

# ─────────── Save ───────────
outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
for (p, stem) in zip((p1, p2),
                     ("heatmap_honest_N25", "heatmap_honest_over_Nramp"))
    savefig(p, joinpath(outdir, stem * ".png"))
    savefig(p, joinpath(outdir, stem * ".pdf"))
    println("Saved: ", joinpath(outdir, stem * ".{png,pdf}"))
end
