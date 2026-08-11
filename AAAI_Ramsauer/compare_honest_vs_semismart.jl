#=
Compare data boundary from the honest MC sampler (no truncation, all M patterns)
against the semismart truncated sampler at the same M = 4.4e6 budget.

Both files use the same N-ramp policy. Column conventions differ slightly:
  semismart: alpha,T,N_used,K_retained,disorder,phi_a,phi_b,q12,phi_max_retained
  honest:    alpha,T,N_used,disorder,phi_a,phi_b,q12,phi_max_other

Output:
  - Console: max |T_bd_honest - T_bd_semismart| in common α range, plus per-α table.
  - panels_paper/compare_honest_vs_semismart.{png,pdf}: two-panel figure.
=#

using Printf
using Plots
using LaTeXStrings
using Statistics

default(dpi = 250)

const φ_star = (sqrt(5) - 1) / 2
const g_max  = 0.5 * log(1 - φ_star^2) + φ_star
phi_eq(T) = 0.5 * (-T + sqrt(T^2 + 4))

# Phi columns by source: (idx_a, idx_b)
const COL_HONEST    = (5, 6)
const COL_SEMISMART = (6, 7)

function read_dataset(csv_path, phi_cols)
    ia, ib = phi_cols
    cells = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
    open(csv_path, "r") do f
        for line in eachline(f)
            startswith(line, "#") && continue
            startswith(line, "alpha") && continue
            fs = split(line, ",")
            α  = parse(Float64, fs[1])
            T  = parse(Float64, fs[2])
            φa = parse(Float64, fs[ia])
            φb = parse(Float64, fs[ib])
            key = (α, T)
            haskey(cells, key) || (cells[key] = Float64[])
            push!(cells[key], φa, φb)
        end
    end
    αs = sort(unique(k[1] for k in keys(cells)))
    Ts = sort(unique(k[2] for k in keys(cells)))
    Z  = fill(NaN, length(Ts), length(αs))
    for (iα, α) in enumerate(αs), (iT, T) in enumerate(Ts)
        (haskey(cells, (α, T)) && !isempty(cells[(α, T)])) || continue
        Z[iT, iα] = mean(cells[(α, T)]) - phi_eq(T)
    end
    return αs, Ts, Z
end

function median_smooth_T(Z, w)
    w <= 1 && return Z
    h = (w - 1) ÷ 2
    nT, nα = size(Z)
    out = copy(Z)
    for iα in 1:nα, iT in 1:nT
        lo = max(1, iT - h); hi = min(nT, iT + h)
        window = Float64[]
        for k in lo:hi
            v = Z[k, iα]
            isnan(v) || push!(window, v)
        end
        isempty(window) || (out[iT, iα] = median(window))
    end
    return out
end

function smooth_vec(v, w)
    w <= 1 && return copy(v)
    h = (w - 1) ÷ 2
    n = length(v)
    out = similar(v)
    for i in 1:n
        lo = max(1, i - h); hi = min(n, i + h)
        out[i] = mean(view(v, lo:hi))
    end
    return out
end

function extract_boundary(αs, Ts, Z, th; smooth_T=3, smooth_bd=5, monotone=true)
    Z_bd = smooth_T > 1 ? median_smooth_T(Z, smooth_T) : Z
    α_bd = Float64[]
    T_bd = Float64[]
    for (iα, α) in enumerate(αs)
        for (iT, T) in enumerate(Ts)
            if !isnan(Z_bd[iT, iα]) && Z_bd[iT, iα] < -th
                push!(α_bd, α)
                push!(T_bd, T)
                break
            end
        end
    end
    smooth_bd > 1 && length(T_bd) > 1 && (T_bd = smooth_vec(T_bd, smooth_bd))
    if monotone && length(T_bd) > 1
        for i in 2:length(T_bd)
            T_bd[i] = min(T_bd[i], T_bd[i-1])
        end
    end
    return α_bd, T_bd
end

# ─── Datasets ────────────────────────────────────────────────────────────
csv_h = joinpath(@__DIR__, "basin_stab_LSE_honest_AAAI_Nramp.csv")
csv_s = joinpath(@__DIR__,
    "basin_stab_LSE_semismart_AAAI_Nramp_M4.4e+06_phikeep0.40.csv")

αh, Th, Zh = read_dataset(csv_h, COL_HONEST)
αs, Ts, Zs = read_dataset(csv_s, COL_SEMISMART)

@printf("honest:    α %d × T %d   range α [%.3f .. %.3f]\n",
        length(αh), length(Th), first(αh), last(αh))
@printf("semismart: α %d × T %d   range α [%.3f .. %.3f]\n",
        length(αs), length(Ts), first(αs), last(αs))

const TH = 0.01
αbd_h, Tbd_h = extract_boundary(αh, Th, Zh, TH)
αbd_s, Tbd_s = extract_boundary(αs, Ts, Zs, TH)

# Restrict to common α range (intersect, matching values to within 1e-9)
α_common = sort(intersect(round.(αbd_h, digits=4), round.(αbd_s, digits=4)))
@printf("Common α grid: %d points [%.3f .. %.3f]\n",
        length(α_common), first(α_common), last(α_common))

# Lookup tables
T_h_map = Dict(round(α, digits=4) => T for (α, T) in zip(αbd_h, Tbd_h))
T_s_map = Dict(round(α, digits=4) => T for (α, T) in zip(αbd_s, Tbd_s))

println()
@printf("  α       T_honest    T_semismart    ΔT\n")
@printf(" -------  ----------  -----------  --------\n")
function summarize(α_common, T_h_map, T_s_map)
    max_dT  = 0.0
    sum_dT2 = 0.0
    for α in α_common
        Th_α = T_h_map[α]
        Ts_α = T_s_map[α]
        ΔT   = Th_α - Ts_α
        @printf("  %.3f   %.5f      %.5f    %+.5f\n", α, Th_α, Ts_α, ΔT)
        max_dT   = max(max_dT, abs(ΔT))
        sum_dT2 += ΔT^2
    end
    return max_dT, sqrt(sum_dT2 / length(α_common))
end
max_dT, rms_dT = summarize(α_common, T_h_map, T_s_map)

println()
@printf("max |ΔT| = %.4f   ;   rms ΔT = %.4f   ;   T grid step ≈ %.4f\n",
        max_dT, rms_dT, length(Th) > 1 ? Th[2]-Th[1] : NaN)

# ─── Plot ────────────────────────────────────────────────────────────────
# Heatmap of honest run; overlay both boundaries; restrict α-range to honest data
α_max_plot = last(αh)
p = heatmap(αs, Ts, -Zs;
            xlabel = L"\alpha", ylabel = L"T",
            c = :Reds, clims = (0.0, 1.0),
            colorbar_title = L"\varphi_{\rm eq}(T) - \langle \varphi \rangle",
            framestyle = :box,
            tick_direction = :out,
            titlefontsize = 10,
            guidefontsize = 10, tickfontsize = 8, legendfontsize = 7,
            title = "M = 4.4e6: honest vs truncated, bd = 0.01",
            size = (700, 480),
            xlims = (0.20, α_max_plot + 0.005),
            ylims = (0.0, 0.5),
            left_margin = 4Plots.mm, bottom_margin = 3Plots.mm)

plot!(p, αbd_s, Tbd_s; color = :red,   lw = 2.2, ls = :solid,
      label = "truncated (M=4.4e6, φ_keep=0.40)")
plot!(p, αbd_h, Tbd_h; color = :black, lw = 2.0, ls = :dash,
      label = "honest (no truncation, M=4.4e6)")

outdir = joinpath(@__DIR__, "panels_paper")
isdir(outdir) || mkpath(outdir)
out_png = joinpath(outdir, "compare_honest_vs_semismart.png")
out_pdf = joinpath(outdir, "compare_honest_vs_semismart.pdf")
savefig(p, out_png); savefig(p, out_pdf)
println("\nSaved: $out_png")
