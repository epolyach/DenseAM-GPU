#=
Kramers validation:
  Left:  log₁₀(τ_data) vs α for fixed T
  Right: log₁₀(τ_data × K) vs ΔF/T — should be linear with slope c
Output: panels_paper/tau_vs_alpha.{png,pdf}, panels_paper/tauK_vs_barrier.{png,pdf}
=#

using Plots, Printf, LaTeXStrings, SpecialFunctions, LinearAlgebra
default(guidefontsize=8, tickfontsize=7, legendfontsize=6)

const FIG_DPI = 300
const FIG_W = round(Int, 86 / 25.4 * 100)
const FIG_H = FIG_W
const M_PAT = 20000
const b_lsr = 2 + sqrt(2)
const φ_c = (b_lsr - 1) / b_lsr

function φ_eq_LSR(T)
    T < 1e-10 && return 1.0
    φ = 0.95
    for _ in 1:200
        D = 1 - b_lsr + b_lsr*φ; D ≤ 1e-10 && (φ = φ_c + 0.005; continue)
        f = (1 - φ^2) - T*φ*D; fp = -2φ - T*(D + b_lsr*φ)
        φ = clamp(φ - f/fp, φ_c + 1e-8, 1 - 1e-8)
    end; return φ
end

function φ_min_val(T)
    φeq = φ_eq_LSR(T)
    return φ_c * φeq - sqrt((1 - φ_c^2) * (1 - φeq^2))
end

function compute_K(α, T)
    N = log(M_PAT) / α; φeq = φ_eq_LSR(T)
    pm = max(0.0, φ_min_val(T))
    logC = loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)
    n_pts = 1000; dφ = (1.0 - pm) / n_pts; dφ ≤ 0 && return 0.0
    integral = 0.0
    for i in 0:n_pts
        φ = pm + i * dφ; φ ≥ 1 && break
        s = 1 - φ^2; s ≤ 0 && continue
        integral += exp(logC + (N-3)/2 * log(s)) * dφ
    end
    return (M_PAT - 1) * integral
end

function barrier_at_phi1max(α, T)
    N = log(M_PAT) / α; φeq = φ_eq_LSR(T)
    R2 = 1 - φeq^2; R2 ≤ 0 && return Inf
    φ1max = sqrt(1 - exp(-2α))
    v_entry = (φ_c - φeq * φ1max) / sqrt(1 - φ1max^2)
    v_entry ≤ 0 && return 0.0
    v2R2 = v_entry^2 / R2
    v2R2 ≥ 0.9999 && return Inf
    return (N-3)/2 * (-log(1 - v2R2))
end

# D_v from v13
v13_lines = readlines(joinpath(@__DIR__, "v13_diffusion.csv"))
v13_α = Float64[]; v13_T = Float64[]; v13_Dv = Float64[]
for line in v13_lines[2:end]
    f = split(line, ","); length(f) < 6 && continue
    push!(v13_α, parse(Float64, f[1])); push!(v13_T, parse(Float64, f[2]))
    push!(v13_Dv, parse(Float64, f[6]))
end
v13_alphas = sort(unique(v13_α))

function interp_Dv(α_q, T_q)
    _, ia = findmin(abs.(v13_alphas .- α_q))
    mask = v13_α .== v13_alphas[ia]
    Ts = v13_T[mask]; Ds = v13_Dv[mask]
    p = sortperm(Ts); Ts = Ts[p]; Ds = Ds[p]
    Dv_peak, i_peak = findmax(Ds)
    T_q > Ts[i_peak] && return Dv_peak
    T_q = clamp(T_q, Ts[1], Ts[i_peak])
    for i in 1:length(Ts)-1
        Ts[i] ≤ T_q ≤ Ts[i+1] || continue
        return Ds[i] + (T_q-Ts[i])/(Ts[i+1]-Ts[i])*(Ds[i+1]-Ds[i])
    end; return Ds[end]
end

function tau_rel(α, T)
    φeq = φ_eq_LSR(T); R2 = 1 - φeq^2
    Dv = interp_Dv(α, T)
    return R2 / Dv
end

# Load τ_data from all files
function load_tau_data()
    dir = @__DIR__
    file_map = Dict{Tuple{Float64,Float64}, String}()
    priority = Dict("v17" => 3, "v16b" => 2, "v16" => 1)
    for f in readdir(dir)
        for pfx in ["v17_Pesc_a", "v16b_Pesc_a", "v16_Pesc_a"]
            if startswith(f, pfx) && endswith(f, ".csv")
                m_a = match(r"a(\d+\.\d+)", f); m_T = match(r"T(\d+\.\d+)", f)
                m_a === nothing && continue; m_T === nothing && continue
                α = parse(Float64, m_a.captures[1]); T = parse(Float64, m_T.captures[1])
                α < 0.20 && continue
                ver = pfx[1:end-7]  # strip _Pesc_a
                key = (α, T)
                old_ver = haskey(file_map, key) ? split(file_map[key],"_")[1] : ""
                if !haskey(file_map, key) || priority[ver] > get(priority, old_ver, 0)
                    file_map[key] = f
                end
            end
        end
    end

    results = Tuple{Float64,Float64,Float64}[]  # (α, T, τ)
    for ((α, T), f) in file_map
        lines = readlines(joinpath(dir, f))
        steps = Float64[]; pesc = Float64[]
        for l in lines[2:end]
            parts = split(l, ","); length(parts) < 2 && continue
            push!(steps, parse(Float64, parts[1])); push!(pesc, parse(Float64, parts[2]))
        end
        surv = max.(1.0 .- pesc, 1e-10)
        # t0
        t0 = 0.0
        for p_hi in [0.10, 0.20, 0.35, 0.50]
            mask = (pesc .> 0.003) .& (pesc .< p_hi)
            sum(mask) < 4 && continue
            t = steps[mask]; y = -log.(1.0 .- pesc[mask])
            X = hcat(ones(length(t)), t); β = X \ y
            if β[2] > 0; t0 = max(0.0, -β[1]/β[2]); break; end
        end
        # Poisson fit
        mask_fit = (surv .> 0.005) .& (steps .> t0)
        sum(mask_fit) < 5 && continue
        t_m = steps[mask_fit]; lnS = log.(surv[mask_fit])
        tau_guess = steps[end] / 5
        best_K = NaN; best_tch = NaN; best_err = Inf
        for lr in range(-2, 2, length=400)
            tch = tau_guess * 10^lr
            u = @. 1 - exp(-(t_m - t0) / tch)
            K = -dot(u, lnS) / dot(u, u)
            K ≤ 0 && continue
            err = sum(@. (lnS + K*u)^2)
            if err < best_err; best_err = err; best_K = K; best_tch = tch; end
        end
        isnan(best_K) && continue
        τ = best_tch / best_K
        push!(results, (α, T, τ))
    end
    return results
end

out_dir = joinpath(@__DIR__, "..", "panels_paper"); mkpath(out_dir)

println("Loading data...")
data = load_tau_data()
@printf("Loaded %d points\n", length(data))

# Print summary table
sort!(data, by=x->(x[1], x[2]))
@printf("\n%-6s %-5s %10s %8s %8s %8s %6s\n", "α", "T", "τ_data", "K", "ΔF/T", "τ_rel", "meta?")
for (α, T, τ) in data
    K = compute_K(α, T); dF = barrier_at_phi1max(α, T)
    τr = tau_rel(α, T); meta = τ > 41000 ? "YES" : "no"
    @printf("%.2f   %.2f  %10.0f  %7.1f  %7.2f  %7.1f  %s\n", α, T, τ, K, dF, τr, meta)
end
println()

# ── Plot 1: log₁₀(τ/K) vs α for fixed T ──
println("Plotting τ/K vs α...")
T_vals = [0.3, 0.5, 0.8, 1.2, 2.0]
colors_T = [:blue, :green, :black, :orange, :red]

p1 = plot(xlabel=L"\alpha", ylabel=L"\log_{10}(\tau / K)",
    xlims=(0.18, 0.36), size=(FIG_W, FIG_H), dpi=FIG_DPI, framestyle=:box, legend=:topright)

for (i, T_target) in enumerate(T_vals)
    αs = Float64[]; τoKs = Float64[]
    for (α, T, τ) in data
        abs(T - T_target) < 0.01 || continue
        K = compute_K(α, T); K < 0.01 && continue
        push!(αs, α); push!(τoKs, τ / K)
    end
    perm = sortperm(αs)
    length(perm) > 0 && plot!(p1, αs[perm], log10.(τoKs[perm]),
        lw=1.5, marker=:circle, markersize=3, color=colors_T[i],
        label=@sprintf("T=%.1f", T_target))
end

for ext in ("png", "pdf")
    savefig(p1, joinpath(out_dir, "tau_vs_alpha.$ext"))
end
println("Saved: panels_paper/tau_vs_alpha.{png,pdf}")

# ── Plot 1b: log₁₀(τ×K) vs α for fixed T ──
println("Plotting τK vs α...")
p1b = plot(xlabel=L"\alpha", ylabel=L"\log_{10}(\tau \times K)",
    xlims=(0.18, 0.36), size=(FIG_W, FIG_H), dpi=FIG_DPI, framestyle=:box, legend=:topright)

for (i, T_target) in enumerate(T_vals)
    αs = Float64[]; τKs = Float64[]
    for (α, T, τ) in data
        abs(T - T_target) < 0.01 || continue
        K = compute_K(α, T); K < 0.01 && continue
        push!(αs, α); push!(τKs, τ * K)
    end
    perm = sortperm(αs)
    length(perm) > 0 && plot!(p1b, αs[perm], log10.(τKs[perm]),
        lw=1.5, marker=:circle, markersize=3, color=colors_T[i],
        label=@sprintf("T=%.1f", T_target))
end

# 1.6/α curve
αs_ref = range(0.18, 0.36, length=100)
plot!(p1b, αs_ref, 1.6 ./ αs_ref,
    lw=2, color=:black, ls=:dash, label=L"1.6/\alpha")

for ext in ("png", "pdf")
    savefig(p1b, joinpath(out_dir, "ktau_vs_alpha.$ext"))
end
println("Saved: panels_paper/ktau_vs_alpha.{png,pdf}")

# ── Plot 1c: log₁₀(τK/τ_rel) and 1.6/α vs α ──
println("Plotting τK/τ_rel and 1.6/α vs α...")
p1c = plot(xlabel=L"\alpha", ylabel=L"\log_{10}(\tau K / \tau_{\mathrm{rel}})",
    xlims=(0.18, 0.36), size=(FIG_W, FIG_H), dpi=FIG_DPI, framestyle=:box, legend=:topright)

for (i, T_target) in enumerate(T_vals)
    αs = Float64[]; ys = Float64[]
    for (α, T, τ) in data
        abs(T - T_target) < 0.01 || continue
        K = compute_K(α, T); K < 0.01 && continue
        τr = tau_rel(α, T)
        push!(αs, α); push!(ys, log10(τ * K / τr))
    end
    perm = sortperm(αs)
    length(perm) > 0 && plot!(p1c, αs[perm], ys[perm],
        lw=1.5, marker=:circle, markersize=3, color=colors_T[i],
        label=@sprintf("T=%.1f", T_target))
end

# 1/α curve
αs_ref = range(0.18, 0.36, length=100)
plot!(p1c, αs_ref, 1.0 ./ αs_ref,
    lw=2, color=:black, ls=:dash, label=L"1/\alpha")

for ext in ("png", "pdf")
    savefig(p1c, joinpath(out_dir, "ktau_tau_rel_vs_alpha.$ext"))
end
println("Saved: panels_paper/ktau_tau_rel_vs_alpha.{png,pdf}")

# ── Plot 1d: ln(Kτ/τ_rel) - c·ΔF/T vs α (c=1) ──
println("Plotting ln(Kτ/τ_rel) - ΔF/T vs α...")
T_all_sorted = sort(unique([T for (_, T, _) in data]))
# Skip T=0.15, T=0.25, T=0.60, T=0.70
T_plot_d = filter(T -> !(T ≈ 0.15 || T ≈ 0.25 || T ≈ 0.60 || T ≈ 0.70), T_all_sorted)

# Distinct colors and line styles (8 T values)
dist_colors = [:red, :blue, :green, :orange, :purple, :black,
               :cyan, :magenta]
dist_styles = [:solid, :solid, :solid, :solid,
               :dash, :dash, :dash, :dash]
dist_markers = [:circle, :square, :diamond, :utriangle,
                :dtriangle, :pentagon, :hexagon, :star5]

c_resid = 1.0
p1d = plot(xlabel=L"\alpha",
    ylabel=L"\ln(\tau K / \tau_{\mathrm{rel}}) - c\,\Delta F/T",
    xlims=(0.18, 0.36), size=(FIG_W + 40, FIG_H), dpi=FIG_DPI, framestyle=:box, legend=:bottomright)

for (i, T_target) in enumerate(T_plot_d)
    αs = Float64[]; ys = Float64[]
    for (α, T, τ) in data
        abs(T - T_target) < 0.01 || continue
        K = compute_K(α, T); K < 0.01 && continue
        dF = barrier_at_phi1max(α, T); isinf(dF) && continue
        τr = tau_rel(α, T)
        push!(αs, α); push!(ys, log(τ * K / τr) - c_resid * dF)
    end
    perm = sortperm(αs)
    length(perm) > 0 && plot!(p1d, αs[perm], ys[perm],
        lw=1.5, ls=dist_styles[i], marker=dist_markers[i], markersize=3,
        color=dist_colors[i],
        label=@sprintf("%.1f", T_target))
end

for ext in ("png", "pdf")
    savefig(p1d, joinpath(out_dir, "residual_vs_alpha.$ext"))
end
println("Saved: panels_paper/residual_vs_alpha.{png,pdf}")

# ── Plot 2: log₁₀(τK/τ_rel) vs ΔF/T ──
println("Plotting τK/τ_rel vs ΔF/T...")
α_vals_unique = sort(unique([α for (α, _, _) in data]))
T_all = sort(unique([T for (_, T, _) in data]))
T_min, T_max = extrema(T_all)

shapes_α = Dict(0.20 => :circle, 0.22 => :square, 0.24 => :diamond,
                0.26 => :utriangle, 0.28 => :dtriangle, 0.30 => :pentagon,
                0.32 => :hexagon, 0.34 => :star5)

# Map T → color via viridis
cmap = cgrad(:viridis)
function T_color(T_val)
    t = clamp((T_val - T_min) / (T_max - T_min), 0, 1)
    return cmap[t]
end

# Colorbar ticks: all T values, labels on a subset
T_labeled = Set([0.2, 0.5, 0.8, 1.2, 2.0])
ctick_vals = collect(T_all)
ctick_strs = [T in T_labeled ? @sprintf("%.1f", T) : "" for T in T_all]

# Layout: main scatter + thin manual colorbar
lay = @layout [a{0.84w} b{0.05w}]

# -- Main scatter panel --
p_main = plot(xlabel=L"\Delta F/T",
    framestyle=:box, legend=:topleft, ylims=(2, 9))

for α_val in α_vals_unique
    xs = Float64[]; ys = Float64[]; Ts = Float64[]
    for (α, T, τ) in data
        abs(α - α_val) < 0.001 || continue
        K = compute_K(α, T); K < 0.01 && continue
        dF = barrier_at_phi1max(α, T); isinf(dF) && continue
        τr = tau_rel(α, T)
        push!(xs, dF); push!(ys, log10(τ * K / τr)); push!(Ts, T)
    end
    length(xs) == 0 && continue
    shape = get(shapes_α, α_val, :circle)
    cols = [T_color(t) for t in Ts]
    scatter!(p_main, xs, ys, markershape=shape,
        markersize=5, markerstrokewidth=0.3, markerstrokecolor=:gray40,
        markercolor=cols, label="")
end

# Legend: dummy scatter for α → shape mapping
for α_val in α_vals_unique
    shape = get(shapes_α, α_val, :circle)
    scatter!(p_main, [NaN], [NaN], markershape=shape,
        markersize=5, color=:gray50, markerstrokewidth=0.3,
        label=@sprintf("%.2f", α_val))
end

# Linear fit to metastable points (τ > 41000)
all_x = Float64[]; all_y = Float64[]
for (α, T, τ) in data
    τ < 41000 && continue
    K = compute_K(α, T); K < 0.01 && continue
    dF = barrier_at_phi1max(α, T); isinf(dF) && continue
    τr = tau_rel(α, T)
    push!(all_x, dF); push!(all_y, log10(τ * K / τr))
end
if length(all_x) > 3
    X = hcat(ones(length(all_x)), all_x)
    β = X \ all_y
    x_range = range(0, maximum(all_x)*1.1, length=100)
    plot!(p_main, x_range, β[1] .+ β[2] .* x_range,
        color=:black, lw=2, ls=:dash,
        label=@sprintf("c=%.2f", β[2]*log(10)))
    @printf("Linear fit: intercept=%.3f → A=%.4f, slope=%.4f → c=%.3f\n",
            β[1], 10.0^(-β[1]), β[2], β[2]*log(10))
end

# c=0.9 reference line (shifted down 1.5 from fit intercept)
if length(all_x) > 3
    slope09 = 0.9 / log(10)
    plot!(p_main, x_range, (β[1] - 1.5) .+ slope09 .* x_range,
        color=:black, lw=1, ls=:dot, label=L"c=0.9")
end

# 1.6/α vs ΔF/T for each data point
for α_val in α_vals_unique
    xs_inv = Float64[]; ys_inv = Float64[]
    for (α, T, τ) in data
        abs(α - α_val) < 0.001 || continue
        K = compute_K(α, T); K < 0.01 && continue
        dF = barrier_at_phi1max(α, T); isinf(dF) && continue
        push!(xs_inv, dF); push!(ys_inv, 1.6 / α)
    end
    length(xs_inv) == 0 && continue
    shape = get(shapes_α, α_val, :circle)
    scatter!(p_main, xs_inv, ys_inv, markershape=shape,
        markersize=4, markerstrokewidth=0.3, markerstrokecolor=:red,
        markercolor=:white, markerstrokealpha=1.0, label="")
end
# Dummy for legend
scatter!(p_main, [NaN], [NaN], markershape=:circle,
    markersize=4, markercolor=:white, markerstrokecolor=:red,
    markerstrokewidth=1, label=L"1.6/\alpha")

# -- Manual colorbar panel --
cb_y = range(T_min, T_max, length=200)
cb_mat = reshape(cb_y, :, 1)
p_cb = heatmap([0], collect(cb_y), cb_mat,
    color=:viridis, colorbar=false, clims=(T_min, T_max),
    xticks=false, framestyle=:box,
    yticks=(ctick_vals, ctick_strs), ymirror=true,
    ylabel=L"T")

p2 = plot(p_main, p_cb, layout=lay,
    size=(FIG_W + 50, FIG_H), dpi=FIG_DPI)

for ext in ("png", "pdf")
    savefig(p2, joinpath(out_dir, "tauK_vs_barrier.$ext"))
end
println("Saved: panels_paper/tauK_vs_barrier.{png,pdf}")
