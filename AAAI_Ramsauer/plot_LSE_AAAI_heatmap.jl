#=
AAAI 2027 — LSE residual heatmap, α ∈ [0.20, 0.70], with three analytical α_c(T) curves.

Reads:
  basin_stab_LSE_honest_AAAI_N25.csv             (honest MC, columns: alpha,T,N_used,disorder,phi_a,phi_b,q12,phi_max_other)
  basin_stab_LSE_semismart_AAAI_N25_phikeep0.40.csv (semismart, columns: alpha,T,N_used,K_retained,disorder,phi_a,phi_b,q12,phi_max_retained)

Writes:
  panels_paper/heatmap_LSE_AAAI_residual_3boundaries.{png,pdf}

The heatmap takes the honest residual where it converged and falls back to the
semismart residual elsewhere; cells with both contribute the average.

Three analytical capacity boundaries are overlaid:
  Gaussian (Ramsauer)             blue solid   T=0 → α_c = 0.5
  Exact-density boundary form     orange dash  T=0 → α_c → ∞
  Exact-density saddle (this work) red solid   T=0 → α_c ≈ 0.6226 (golden)
=#

ENV["GKSwstype"] = "100"
using Plots, Printf, Statistics

const FIG_DPI    = 300
const FIG_W      = round(Int, 86 / 25.4 * 100) + 80
const FIG_H      = round(Int, 86 / 25.4 * 100)
const FONT_GUIDE = 8
const FONT_TICK  = 7
const FONT_LEG   = 6
default(guidefontsize=FONT_GUIDE, tickfontsize=FONT_TICK, legendfontsize=FONT_LEG)

out_dir = joinpath(@__DIR__, "panels_paper")
mkpath(out_dir)

# ─── Loaders ───
struct MCData
    alpha :: Vector{Float64}
    T     :: Vector{Float64}
    phi_a :: Vector{Float64}
    phi_b :: Vector{Float64}
end

# Generic CSV reader that skips comment lines and the header
function load_csv(path::String; phi_a_col::Int, phi_b_col::Int)
    isfile(path) || (@warn "Missing $path — skipping"; return nothing)
    lines = filter(l -> !startswith(l, "#") && !isempty(l), readlines(path))
    lines = lines[2:end]  # drop header
    n = length(lines)
    alpha = zeros(n); T = zeros(n); pa = zeros(n); pb = zeros(n)
    for i in 1:n
        f = split(lines[i], ",")
        alpha[i] = parse(Float64, f[1])
        T[i]     = parse(Float64, f[2])
        pa[i]    = parse(Float64, f[phi_a_col])
        pb[i]    = parse(Float64, f[phi_b_col])
    end
    return MCData(alpha, T, pa, pb)
end

honest_path             = joinpath(@__DIR__, "basin_stab_LSE_honest_AAAI_N25.csv")
semismart_path          = joinpath(@__DIR__, "basin_stab_LSE_semismart_AAAI_N25_phikeep0.40.csv")
nramp_path              = joinpath(@__DIR__, "basin_stab_LSE_semismart_AAAI_Nramp.csv")
honest_nramp_path       = joinpath(@__DIR__, "basin_stab_LSE_honest_AAAI_Nramp.csv")
nramp_m44e6_path        = joinpath(@__DIR__, "basin_stab_LSE_semismart_AAAI_Nramp_M4.4e6.csv")

# Column layouts:
# honest / honest_nramp:  alpha,T,N_used,disorder,phi_a,phi_b,q12,phi_max_other      → phi_a col 5, phi_b col 6
# semismart / nramp / *:  alpha,T,N_used,K_retained,disorder,phi_a,phi_b,q12,phi_max → phi_a col 6, phi_b col 7
honest_data             = load_csv(honest_path;        phi_a_col=5, phi_b_col=6)
semismart_data          = load_csv(semismart_path;     phi_a_col=6, phi_b_col=7)
nramp_data              = load_csv(nramp_path;         phi_a_col=6, phi_b_col=7)
honest_nramp_data       = load_csv(honest_nramp_path;  phi_a_col=5, phi_b_col=6)
nramp_m44e6_data        = load_csv(nramp_m44e6_path;   phi_a_col=6, phi_b_col=7)

all(x -> x === nothing,
    (honest_data, semismart_data, nramp_data, honest_nramp_data, nramp_m44e6_data)) &&
    error("All CSVs missing — nothing to plot.")

# ─── Build common (α, T) grid ───
all_alphas = Float64[]; all_Ts = Float64[]
for d in (honest_data, semismart_data, nramp_data, honest_nramp_data, nramp_m44e6_data)
    d === nothing && continue
    append!(all_alphas, d.alpha); append!(all_Ts, d.T)
end
alphas = sort(unique(round.(all_alphas, digits=4)))
Ts     = sort(unique(round.(all_Ts,     digits=5)))
na = length(alphas); nT = length(Ts)
@printf("Grid: α=%d values [%.2f, %.2f]   T=%d values [%.4f, %.4f]\n",
        na, alphas[1], alphas[end], nT, Ts[1], Ts[end])

# Per-source per-cell mean
function build_phi_grid(d::Union{MCData,Nothing})
    d === nothing && return fill(NaN, nT, na)
    g = fill(NaN, nT, na)
    for ia in 1:na, iT in 1:nT
        mask = (abs.(d.alpha .- alphas[ia]) .< 1e-4) .& (abs.(d.T .- Ts[iT]) .< 1e-5)
        if any(mask)
            g[iT, ia] = mean(vcat(d.phi_a[mask], d.phi_b[mask]))
        end
    end
    return g
end
phi_honest            = build_phi_grid(honest_data)
phi_semismart         = build_phi_grid(semismart_data)
phi_nramp             = build_phi_grid(nramp_data)
phi_honest_nramp      = build_phi_grid(honest_nramp_data)
phi_nramp_m44e6       = build_phi_grid(nramp_m44e6_data)

# Priority (highest → lowest):
#   5 = honest_Nramp           (matched-N ground truth, M_TARGET=4.4e6)
#   4 = semismart_Nramp_M4.4e6 (matched N as honest_Nramp, with truncation)
#   3 = semismart_Nramp        (M_TARGET=4.4e7, larger N at small α)
#   2 = honest_N25             (N=25 ground truth)
#   1 = semismart_N25          (N=25, PHI_KEEP=0.40 baseline)
phi_combined = fill(NaN, nT, na)
src_combined = fill(0,   nT, na)
for i in eachindex(phi_combined)
    if isfinite(phi_honest_nramp[i])
        phi_combined[i] = phi_honest_nramp[i]; src_combined[i] = 5
    elseif isfinite(phi_nramp_m44e6[i])
        phi_combined[i] = phi_nramp_m44e6[i];  src_combined[i] = 4
    elseif isfinite(phi_nramp[i])
        phi_combined[i] = phi_nramp[i];        src_combined[i] = 3
    elseif isfinite(phi_honest[i])
        phi_combined[i] = phi_honest[i];       src_combined[i] = 2
    elseif isfinite(phi_semismart[i])
        phi_combined[i] = phi_semismart[i];    src_combined[i] = 1
    end
end
n5 = count(==(5), src_combined); n4 = count(==(4), src_combined)
n3 = count(==(3), src_combined); n2 = count(==(2), src_combined); n1 = count(==(1), src_combined)
@printf("Source split: honest_Nramp=%d  smart_Nramp_M4.4e6=%d  smart_Nramp=%d  honest_N25=%d  smart_N25=%d  (none=%d)\n",
        n5, n4, n3, n2, n1, length(src_combined) - n5 - n4 - n3 - n2 - n1)

# Residual ⟨φ⟩ − φ_eq(T)
φ_eq(t) = 0.5 * (-t + sqrt(t^2 + 4))
res = similar(phi_combined)
for iT in 1:nT
    base = φ_eq(Ts[iT])
    res[iT, :] .= phi_combined[iT, :] .- base
end
finite_res = filter(isfinite, vec(res))
cmax = max(0.05, isempty(finite_res) ? 0.5 : maximum(abs, finite_res))

# Residual disagreement honest vs semismart in overlap region
overlap_disagree = filter(isfinite, vec(abs.(phi_honest .- phi_semismart)))
if !isempty(overlap_disagree)
    @printf("Honest/semismart overlap cells: %d, max |Δφ| = %.4f, mean |Δφ| = %.4f\n",
            length(overlap_disagree), maximum(overlap_disagree), mean(overlap_disagree))
end

# ─── Three analytical capacity boundaries ───
f_ret(t)     = let φ = φ_eq(t); 1 - φ - (t/2)*log(1 - φ^2); end
α_c_gauss(t) = let fr = f_ret(t); fr >= 1 ? 0.0 : 0.5*(1 - fr)^2; end
function α_c_bd(t)
    fr = f_ret(t); fr >= 1 && return 0.0
    arg = 1 - (1 - fr)^2
    arg <= 0 ? Inf : -0.5*log(arg)
end
const φ_star  = (sqrt(5.0) - 1)/2                             # golden ratio
const G_MAX   = 0.5*log(φ_star) - φ_star^2                    # = -0.6226 (paper notation)
const g_max_p = 0.5*log(φ_star) + φ_star                      # ≈ +0.3774 (saddle value g_max)
α_c_sd(t)     = -G_MAX - f_ret(t)
α_c_sp(t)     = φ_eq(t) - g_max_p                             # single-pattern spinodal

# ─── Plot ───
xmin, xmax = 0.20, 0.70
ymin, ymax = 0.0, 0.5

p = heatmap(alphas, Ts, res,
    color=cgrad(:RdBu), clims=(-cmax, cmax),
    xlabel="α  =  ln M / N", ylabel="T",
    xlims=(xmin, xmax), ylims=(ymin, ymax),
    colorbar_title="⟨φ⟩ − φ_eq(T)",
    title="LSE basin stability  +  4 analytical α_c(T)  (Nramp ⊕ N=25, multi-source)",
    titlefontsize=FONT_GUIDE,
    size=(FIG_W, FIG_H), dpi=FIG_DPI,
    left_margin=2Plots.mm, bottom_margin=2Plots.mm)

T_range = collect(range(max(ymin, 1e-4), ymax, length=600))
α_g  = [α_c_gauss(t) for t in T_range]
α_b  = [α_c_bd(t)    for t in T_range]
α_s  = [α_c_sd(t)    for t in T_range]
α_sp = [α_c_sp(t)    for t in T_range]
m_g  = isfinite.(α_g)  .& (α_g  .>= xmin) .& (α_g  .<= xmax)
m_b  = isfinite.(α_b)  .& (α_b  .>= xmin) .& (α_b  .<= xmax)
m_s  = isfinite.(α_s)  .& (α_s  .>= xmin) .& (α_s  .<= xmax)
m_sp = isfinite.(α_sp) .& (α_sp .>= xmin) .& (α_sp .<= xmax)
plot!(p, α_g[m_g],   T_range[m_g],  color=:blue,       lw=2.0, ls=:solid, label="α_c^Gauss (Ramsauer)")
plot!(p, α_b[m_b],   T_range[m_b],  color=:darkorange, lw=2.0, ls=:dash,  label="α_c^bd  boundary form")
plot!(p, α_s[m_s],   T_range[m_s],  color=:red,        lw=2.0, ls=:solid, label="α_c^sd  saddle (this work)")
plot!(p, α_sp[m_sp], T_range[m_sp], color=:purple,     lw=2.0, ls=:dot,   label="α_c^sp  single-pattern spinodal")

# Empirical spinodal anchors per N: distinct marker per N, points connected
# by dotted lines.
anchors_path = joinpath(@__DIR__, "spinodal_empirical_anchors.csv")
if isfile(anchors_path)
    rows_e = Tuple{Float64,Int,Float64}[]
    for line in eachline(anchors_path)
        (isempty(line) || startswith(line, "alpha") || startswith(line, "#")) && continue
        f = split(line, ",")
        push!(rows_e, (parse(Float64, f[1]), parse(Int, f[2]), parse(Float64, f[3])))
    end
    if !isempty(rows_e)
        Ns_e = sort(unique([r[2] for r in rows_e]))
        marker_for(i, n) = (n == maximum(Ns_e)) ? :star5 :
                           (i == 1) ? :utriangle : :rect
        size_for(n)      = (n == maximum(Ns_e)) ? 8 : 6
        for (i, N) in enumerate(Ns_e)
            pts = sort([(r[1], r[3]) for r in rows_e if r[2] == N])
            αs_N = [p[1] for p in pts]
            Ts_N = [p[2] for p in pts]
            plot!(p, αs_N, Ts_N,
                  color=:purple, lw=1.0, ls=:dot,
                  marker=marker_for(i, N), ms=size_for(N),
                  markerstrokecolor=:black, markerstrokewidth=1,
                  markercolor=:purple,
                  label="α_c^sp  empirical N=$N")
        end
    end
end

for ext in ("png", "pdf")
    savefig(p, joinpath(out_dir, @sprintf("heatmap_LSE_AAAI_residual_3boundaries.%s", ext)))
end
@printf("φ* = %.6f   g_max = %.6f   α_c^sd(T=0) = %.4f\n",
        φ_star, G_MAX, -G_MAX)
@printf("Saved: %s/heatmap_LSE_AAAI_residual_3boundaries.{png,pdf}\n", out_dir)
println("Done.")
