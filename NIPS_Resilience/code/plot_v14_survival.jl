#=
Plot v14 survival curves S(t) = 1 - P_esc(t) with ⟨λ⟩ fits
────────────────────────────────────────────────────────────────────────
For each (α,T): plot S(t) on log scale, overlay exp(-⟨λ⟩ t),
annotate τ = 1/⟨λ⟩.

⟨λ⟩ computed from initial slope of -ln S(t) (early-time regime
before disorder heterogeneity distorts the curve).

Output: panels_paper/v14_survival_all.{png,pdf}
────────────────────────────────────────────────────────────────────────
=#

using Plots, Printf, LaTeXStrings, LinearAlgebra, SpecialFunctions
default(guidefontsize=7, tickfontsize=6, legendfontsize=6)

# ──────────────── Global compound Poisson model ────────────────
const GLOBAL_A = 4.5146
const GLOBAL_C = 0.5051
const M_PAT = 20000
const b_lsr_g = 2 + sqrt(2)
const φ_c_g = (b_lsr_g - 1) / b_lsr_g

function φ_eq_g(T)
    T < 1e-10 && return 1.0
    φ = 0.95
    for _ in 1:200
        D = 1 - b_lsr_g + b_lsr_g*φ
        D ≤ 1e-10 && (φ = φ_c_g + 0.005; continue)
        f = (1 - φ^2) - T*φ*D
        fp = -2φ - T*(D + b_lsr_g*φ)
        φ = clamp(φ - f/fp, φ_c_g + 1e-8, 1 - 1e-8)
    end
    return φ
end

# D_v interpolation from v13
v13_lines_g = readlines(joinpath(@__DIR__, "v13_diffusion.csv"))
const v13_α_g = Float64[]; const v13_T_g = Float64[]; const v13_Dv_g = Float64[]
for line in v13_lines_g[2:end]
    f = split(line, ","); length(f) < 6 && continue
    push!(v13_α_g, parse(Float64, f[1]))
    push!(v13_T_g, parse(Float64, f[2]))
    push!(v13_Dv_g, parse(Float64, f[6]))
end
const v13_alphas_g = sort(unique(v13_α_g))

function interp_Dv_g(α_q, T_q)
    _, ia = findmin(abs.(v13_alphas_g .- α_q))
    α_near = v13_alphas_g[ia]
    mask = v13_α_g .== α_near
    Ts = v13_T_g[mask]; Ds = v13_Dv_g[mask]
    p = sortperm(Ts); Ts = Ts[p]; Ds = Ds[p]
    T_q = clamp(T_q, Ts[1], Ts[end])
    for i in 1:length(Ts)-1
        if Ts[i] ≤ T_q ≤ Ts[i+1]
            frac = (T_q - Ts[i]) / (Ts[i+1] - Ts[i])
            return Ds[i] + frac * (Ds[i+1] - Ds[i])
        end
    end
    return Ds[end]
end

function compound_poisson_S(t_arr, α, T, N)
    φeq = φ_eq_g(T)
    R2 = 1 - φeq^2; R2 ≤ 1e-10 && return ones(length(t_arr))
    Dv = interp_Dv_g(α, T)
    τ_rel = R2 / Dv
    rate_pre = GLOBAL_A / τ_rel

    φ_min = max(0.0, φ_c_g * (φeq - sqrt(R2))) + 1e-8
    φ_zb = φ_c_g / φeq
    φ_hi = min(φ_zb, 0.9999)
    logC = loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)

    # Precompute channel rates and densities
    λ_list = Float64[]; f_list = Float64[]; dφ_list = Float64[]
    if φ_hi > φ_min
        n_φ = 500; dφ = (φ_hi - φ_min) / n_φ
        for i in 0:n_φ
            φ = φ_min + i * dφ
            s = 1 - φ^2; s ≤ 0 && continue
            v = (φ_c_g - φeq*φ) / sqrt(s)
            fv = exp(logC + (N-3)/2*log(s))
            if v ≤ 0
                push!(λ_list, rate_pre); push!(f_list, fv); push!(dφ_list, dφ)
            else
                v2R2 = v^2/R2; v2R2 ≥ 0.9999 && continue
                ΔFT = (N-3)/2 * (-log(1 - v2R2))
                push!(λ_list, rate_pre * exp(-GLOBAL_C * ΔFT))
                push!(f_list, fv); push!(dφ_list, dφ)
            end
        end
    end
    # Barrierless region
    if φ_zb < 0.9999
        dφ2 = (0.9999 - φ_zb) / 100
        for i in 0:100
            φ = φ_zb + i * dφ2; s = 1-φ^2; s ≤ 0 && continue
            push!(λ_list, rate_pre)
            push!(f_list, exp(logC + (N-3)/2*log(s)))
            push!(dφ_list, dφ2)
        end
    end

    S_out = ones(length(t_arr))
    for (it, t) in enumerate(t_arr)
        integral = 0.0
        for j in eachindex(λ_list)
            integral += (1 - exp(-λ_list[j] * t)) * f_list[j] * dφ_list[j]
        end
        S_out[it] = exp(-(M_PAT - 1) * integral)
    end
    return S_out
end

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

# ──────────────── Collect all v14 files ────────────────
v14_dir = @__DIR__
all_files = filter(f -> startswith(f, "v14_Pesc_a") && endswith(f, ".csv"), readdir(v14_dir))

# Parse (α, T) from filenames
data = Dict{Tuple{Float64,Float64}, Vector{Tuple{Int,Float64}}}()
for f in all_files
    m_a = match(r"a(\d+\.\d+)", f)
    m_T = match(r"T(\d+\.\d+)", f)
    m_a === nothing && continue; m_T === nothing && continue
    α = parse(Float64, m_a.captures[1])
    T = parse(Float64, m_T.captures[1])
    rows = Tuple{Int,Float64}[]
    lines = readlines(joinpath(v14_dir, f))
    for line in lines[2:end]
        parts = split(line, ",")
        length(parts) < 2 && continue
        push!(rows, (parse(Int, parts[1]), parse(Float64, parts[2])))
    end
    data[(α, T)] = rows
end

alphas = sort(unique([k[1] for k in keys(data)]))
Ts_all = sort(unique([k[2] for k in keys(data)]))

na = length(alphas); nT = length(Ts_all)
@printf("v14 data: %d α × %d T = %d panels\n", na, nT, length(data))

# ──────────────── Compute ⟨λ⟩ from initial slope ────────────────
function fit_mean_rate(steps, pesc; T_val=0.0)
    # Adaptive fit:
    #   T ≥ 0.30: wide range (P up to 0.90) — decay is near-exponential
    #   T < 0.30: narrow range (P up to 0.20) — initial slope only,
    #             heavy tail from disorder heterogeneity distorts wide fit
    p_hi_list = T_val ≥ 0.30 ? (0.90, 0.80, 0.60, 0.40) : (0.20, 0.30, 0.50, 0.80)

    for p_hi in p_hi_list
        mask = (pesc .> 0.005) .& (pesc .< p_hi)
        if sum(mask) >= 4
            t = Float64.(steps[mask])
            y = -log.(1.0 .- pesc[mask])
            n = length(t)
            X = hcat(ones(n), t)
            β = X \ y       # β = [b, λ]
            λ = β[2]
            λ ≤ 0 && continue
            t0 = max(0.0, -β[1] / λ)
            return λ, t0
        end
    end
    return NaN, 0.0
end

# ──────────────── Plot grid ────────────────
println("Plotting survival grid...")
plots_arr = []

# Rows: T decreasing (top = highest T), Columns: α increasing
Ts_desc = reverse(Ts_all)

for iT in 1:nT
    for ia in 1:na
        T_val = Ts_desc[iT]
        α_val = alphas[ia]
        key = (α_val, T_val)

        if haskey(data, key)
            rows = data[key]
            steps = [r[1] for r in rows]
            pesc  = [r[2] for r in rows]
            surv  = 1.0 .- pesc  # S(t) = 1 - P_esc(t)

            # Step 1: get lag t₀ from quick slope fit
            _, t0 = fit_mean_rate(steps, pesc; T_val=T_val)

            t_max = steps[end]
            t_plot = Float64.(steps)
            s_plot = max.(surv, 1e-4)

            # x-range: where S(t) drops to ~1e-3 (or end of data)
            idx_end = findfirst(s -> s < 1e-3, surv)
            t_xmax = idx_end !== nothing ? Float64(steps[idx_end]) * 1.1 : Float64(t_max)

            # Step 2: fit Poisson model S(t) = exp(-K(1-exp(-(t-t₀)/τ_ch)))
            # This determines BOTH the magenta curve AND the red slope.
            mask_fit = (surv .> 0.005) .& (Float64.(steps) .> t0)
            best_K = NaN; best_τch = NaN
            λ_quick, _ = fit_mean_rate(steps, pesc; T_val=T_val)
            τ_guess = isnan(λ_quick) || λ_quick ≤ 0 ? Float64(t_max)/5 : 1.0/λ_quick

            if sum(mask_fit) >= 5
                t_m = Float64.(steps[mask_fit])
                lnS_m = log.(surv[mask_fit])
                best_err = Inf
                for log_r in range(-2.0, 2.0, length=400)
                    τ_try = τ_guess * 10.0^log_r
                    u = [1.0 - exp(-(t - t0) / τ_try) for t in t_m]
                    K_try = -dot(u, lnS_m) / dot(u, u)
                    K_try ≤ 0 && continue
                    pred = [-K_try * ui for ui in u]
                    err = sum((lnS_m .- pred).^2)
                    if err < best_err
                        best_err = err; best_K = K_try; best_τch = τ_try
                    end
                end
            end

            # Derive ⟨λ⟩ from Poisson: initial slope = K/τ_ch
            λ_mean = (!isnan(best_K) && best_K > 0) ? best_K / best_τch : NaN
            τ_val = isnan(λ_mean) || λ_mean ≤ 0 ? NaN : 1.0 / λ_mean

            N_val = max(round(Int, log(20000) / α_val), 2)

            p = plot(t_plot, s_plot,
                     yscale=:log10, ylims=(1e-3, 1.0),
                     xlims=(0, t_xmax),
                     lw=1.2, color=:steelblue, label=false,
                     xlabel="t", ylabel=(ia == 1 ? "S(t)" : ""),
                     title=@sprintf("α=%.2f T=%.2f N=%d", α_val, T_val, N_val),
                     titlefontsize=7, framestyle=:box,
                     xrotation=30, xguidefontsize=6,
                     left_margin=(ia==1 ? 5Plots.mm : 1Plots.mm),
                     bottom_margin=4Plots.mm)

            if !isnan(λ_mean) && λ_mean > 0
                t_fit = range(0, t_xmax, length=300)

                # Red dashed: exp(-⟨λ⟩(t-t₀)) — tangent of Poisson at t₀
                s_exp = [t > t0 ? exp(-λ_mean * (t - t0)) : 1.0 for t in t_fit]
                plot!(p, t_fit, s_exp, lw=1.0, ls=:dash, color=:red, label=false)

                # Magenta dotted: Poisson model
                s_poiss = [t > t0 ?
                    exp(-best_K * (1 - exp(-(t - t0) / best_τch))) : 1.0
                    for t in t_fit]
                plot!(p, t_fit, s_poiss, lw=1.2, ls=:dot, color=:magenta, label=false)

                # Green solid: global compound Poisson (A=4.5, c=0.5)
                s_global = compound_poisson_S(collect(t_fit), α_val, T_val, N_val)
                plot!(p, t_fit, s_global, lw=1.5, ls=:solid, color=:green, label=false)

                # Annotate τ, K, τ_ch
                if τ_val < 1e6
                    τ_str = τ_val ≥ 1000 ? @sprintf("τ=%.0fk", τ_val/1000) : @sprintf("τ=%.0f", τ_val)
                else
                    τ_str = @sprintf("τ=%.0ek", τ_val/1000)
                end
                K_str = @sprintf("K=%.1f", best_K)
                τch_str = best_τch ≥ 1000 ? @sprintf("τ_ch=%.0fk", best_τch/1000) : @sprintf("τ_ch=%.0f", best_τch)
                ann = τ_str * "\n" * K_str * " " * τch_str
                annotate!(p, t_xmax*0.95, 0.3, text(ann, :red, 6, :right))
            end
        else
            # Empty panel
            p = plot(framestyle=:box, grid=false,
                     title=@sprintf("α=%.2f T=%.2f", alphas[ia], Ts_desc[iT]),
                     titlefontsize=7)
            annotate!(p, 0.5, 0.5, text("—", :gray, 10, :center))
        end

        push!(plots_arr, p)
    end
end

# Arrange: rows = T (top = highest T), columns = α
fig = plot(plots_arr..., layout=(nT, na),
           size=(na * 250, nT * 260), dpi=200,
           margin=2Plots.mm)

for ext in ("png", "pdf")
    savefig(fig, joinpath(out_dir, "v14_survival_all.$ext"))
end
println("Saved: panels_paper/v14_survival_all.{png,pdf}")
