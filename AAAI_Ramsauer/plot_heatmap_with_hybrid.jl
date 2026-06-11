#=
Joint heatmaps overlaying the 1D hybrid MC data on the existing
honest-N25 + semismart-Nramp base.

Sources, priority high → low:
  1. hybrid_1d_LSE_N{N}.csv  (this work, fixed N, 1D MC)         α ∈ [0.50, 0.62]
  2. basin_stab_LSE_honest_AAAI_N25.csv                          α ∈ [0.20, 0.56]
  3. basin_stab_LSE_semismart_AAAI_Nramp_M4.4e6.csv              α ∈ [0.20, 0.70]

Outputs:
  panels_paper/heatmap_honest_over_Nramp_N50.png/pdf
  panels_paper/heatmap_honest_over_Nramp_N100.png/pdf
=#

using Printf
using Plots
using Statistics

const φ_star = (sqrt(5) - 1) / 2
const g_max  = 0.5 * log(φ_star) + φ_star          # ≈ 0.3774

phi_eq(T)   = 0.5 * (-T + sqrt(T^2 + 4))
fret(T)     = (1 - phi_eq(T)) - 0.5 * T * log(1 - phi_eq(T)^2)
ac_gauss(T) = 0.5 * (1 - fret(T))^2
ac_bd(T)    = -0.5 * log(1 - (1 - fret(T))^2)
ac_sd(T)    = 1 - g_max - fret(T)
ac_sp(T)    = phi_eq(T) - g_max

# Load CSV: returns Dict{(α,T)} → mean φ
function load_phi_mean(path; phi_cols::Vector{Int})
    cells = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
    isfile(path) || return cells
    open(path, "r") do f
        for line in eachline(f)
            (isempty(line) || startswith(line, "#") || startswith(line, "alpha") || startswith(line, "generator")) && continue
            fs = split(line, ",")
            length(fs) <= maximum(phi_cols) && continue
            α = parse(Float64, fs[1])
            T = parse(Float64, fs[2])
            key = (round(α, digits=4), round(T, digits=5))
            haskey(cells, key) || (cells[key] = Float64[])
            for j in phi_cols
                push!(cells[key], parse(Float64, fs[j]))
            end
        end
    end
    return cells
end

# Find nearest base T (step 0.01 centered at 0.005) for a given T
base_T_for(T) = 0.005 + 0.01 * round((T - 0.005) / 0.01)

const HONEST_CSV   = joinpath(@__DIR__, "basin_stab_LSE_honest_AAAI_N25.csv")
const SEMI_CSV     = joinpath(@__DIR__, "basin_stab_LSE_semismart_AAAI_Nramp_M4.4e6.csv")

# honest:  α,T,N_used,disorder,phi_a,phi_b,...                    cols 5, 6
# semismart: α,T,N_used,K_retained,disorder,phi_a,phi_b,...       cols 6, 7
# hybrid:  α,T,disorder,K,phi_mean,...                            col 5
cells_honest = load_phi_mean(HONEST_CSV;   phi_cols=[5, 6])
cells_semi   = load_phi_mean(SEMI_CSV;     phi_cols=[6, 7])

# Per-α list of (T, φ) data points, hybrid first then base (highest priority wins
# when T values coincide). The list is sorted by T.
function column_points(α, cells_hybrid, Ts_hybrid, Ts_base)
    αk = round(α, digits=4)
    points = Tuple{Float64,Float64}[]
    used = Set{Float64}()
    for T in Ts_hybrid
        key = (αk, round(T, digits=5))
        if haskey(cells_hybrid, key) && !isempty(cells_hybrid[key])
            push!(points, (T, mean(cells_hybrid[key])))
            push!(used, round(T, digits=5))
        end
    end
    for T in Ts_base
        Tk = round(T, digits=5)
        Tk in used && continue
        key = (αk, Tk)
        v = if haskey(cells_honest, key) && !isempty(cells_honest[key])
            mean(cells_honest[key])
        elseif haskey(cells_semi, key) && !isempty(cells_semi[key])
            mean(cells_semi[key])
        else
            NaN
        end
        if !isnan(v)
            push!(points, (T, v))
            push!(used, Tk)
        end
    end
    sort!(points; by = first)
    return points
end

# Linear interpolation. Edges clamp to nearest data point.
function lin_interp(points::Vector{Tuple{Float64,Float64}}, T)
    isempty(points) && return NaN
    n = length(points)
    points[1][1] >= T && return points[1][2]
    points[n][1] <= T && return points[n][2]
    for i in 2:n
        if points[i][1] >= T
            t1, v1 = points[i-1]
            t2, v2 = points[i]
            return v1 + (v2 - v1) * (T - t1) / (t2 - t1)
        end
    end
    return points[n][2]
end

function build_combined(cells_hybrid, αs, Ts; Ts_hybrid, Ts_base)
    Z = fill(NaN, length(Ts), length(αs))
    for (iα, α) in enumerate(αs)
        pts = column_points(α, cells_hybrid, Ts_hybrid, Ts_base)
        isempty(pts) && continue
        for (iT, T) in enumerate(Ts)
            v = lin_interp(pts, T)
            isnan(v) || (Z[iT, iα] = v - phi_eq(T))
        end
    end
    return Z
end

function make_plot(Z, αs, Ts, title; rect=nothing)
    p = heatmap(αs, Ts, Z;
                xlabel = "α", ylabel = "T", title = title,
                c = :RdBu, clims = (-0.9, 0.9),
                colorbar_title = "⟨φ⟩ − φ_eq(T)",
                size = (940, 540), framestyle = :box,
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
    if rect !== nothing
        x1, x2, y1, y2 = rect
        plot!(p, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
              color = RGBA(0.5, 0.5, 0.5, 0.9), lw = 1.5, ls = :solid,
              label = "hybrid MC zone")
    end
    xlims!(p, (0.20, 0.70))
    ylims!(p, (0.0, 0.5))
    return p
end

function build_grid()
    αs = sort(unique(vcat(
        [k[1] for k in keys(cells_honest)],
        [k[1] for k in keys(cells_semi)])))
    Ts = sort(unique(vcat(
        [k[2] for k in keys(cells_honest)],
        [k[2] for k in keys(cells_semi)])))
    return αs, Ts
end

αs_base, Ts_base = build_grid()

# Hybrid α gets added if missing; T grid is uniform at the finer hybrid step (0.005)
# throughout — base data is linearly interpolated where it is coarser.
αs_hybrid_grid = collect(0.50:0.01:0.62)
Ts_hybrid_grid = collect(0.0025:0.005:0.0975)
αs_all = sort(unique(vcat(αs_base, αs_hybrid_grid)))
Ts_fine = collect(0.0025:0.005:0.4975)

# Hybrid rectangle outline for the plot
const HYB_RECT = (αs_hybrid_grid[1] - 0.005, αs_hybrid_grid[end] + 0.005,
                  0.0, Ts_hybrid_grid[end] + 0.0025)

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)

for N in (50, 100)
    hyb_path = joinpath(@__DIR__, @sprintf("hybrid_1d_LSE_N%d.csv", N))
    if !isfile(hyb_path)
        @warn "Missing $hyb_path — skip"
        continue
    end
    cells_hyb = load_phi_mean(hyb_path; phi_cols=[5])
    if isempty(cells_hyb)
        @warn "No hybrid data in $hyb_path"
        continue
    end
    Z = build_combined(cells_hyb, αs_all, Ts_fine;
                       Ts_hybrid=Ts_hybrid_grid, Ts_base=Ts_base)
    title = @sprintf("⟨φ⟩ − φ_eq(T), hybrid MC N=%d (priority) ⊕ honest N=25 ⊕ semismart Nramp", N)
    p = make_plot(Z, αs_all, Ts_fine, title; rect=HYB_RECT)
    stem = @sprintf("heatmap_honest_over_Nramp_N%d", N)
    out_png = joinpath(outdir, stem * ".png")
    out_pdf = joinpath(outdir, stem * ".pdf")
    out_eps = joinpath(outdir, stem * ".eps")
    savefig(p, out_png); savefig(p, out_pdf)
    run(`pdftops -eps $out_pdf $out_eps`)
    println("Saved: ", out_png, "  ", out_pdf, "  ", out_eps)
end
