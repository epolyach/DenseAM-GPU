#=
Plot (A, c) calibration diagnostics:
  Left:  MSE landscape in (log₁₀A, c) space
  Right: τ_model vs τ_data scatter (log-log)

Also compute error estimates for A, c from the Hessian at the minimum.

Output: panels_paper/Ac_calibration.{png,pdf}
=#

using Plots, Printf, LaTeXStrings, LinearAlgebra, SpecialFunctions, Statistics
default(guidefontsize=8, tickfontsize=7, legendfontsize=6)

out_dir = joinpath(@__DIR__, "..", "panels_paper")
mkpath(out_dir)

const M_PAT = 20000
const b_lsr_val = 2 + sqrt(2)
const φ_c_val = (b_lsr_val - 1) / b_lsr_val

function φ_eq_LSR(T)
    b = b_lsr_val; pc = φ_c_val
    T < 1e-10 && return 1.0
    φ = 0.95
    for _ in 1:200
        D = 1 - b + b*φ
        D ≤ 1e-10 && (φ = pc + 0.005; continue)
        f = (1 - φ^2) - T*φ*D
        fp = -2φ - T*(D + b*φ)
        φ = clamp(φ - f/fp, pc + 1e-8, 1 - 1e-8)
    end
    return φ
end

function log_C_N(N)
    loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)
end

# ── D_v interpolation ──
v13_lines = readlines(joinpath(@__DIR__, "v13_diffusion.csv"))
v13_α = Float64[]; v13_T = Float64[]; v13_Dv = Float64[]
for line in v13_lines[2:end]
    f = split(line, ","); length(f) < 6 && continue
    push!(v13_α, parse(Float64, f[1]))
    push!(v13_T, parse(Float64, f[2]))
    push!(v13_Dv, parse(Float64, f[6]))
end
v13_alphas = sort(unique(v13_α))

function interp_Dv(α_q, T_q)
    _, ia = findmin(abs.(v13_alphas .- α_q))
    α_near = v13_alphas[ia]
    mask = v13_α .== α_near
    Ts = v13_T[mask]; Ds = v13_Dv[mask]
    p = sortperm(Ts); Ts = Ts[p]; Ds = Ds[p]
    Dv_peak, i_peak = findmax(Ds)
    T_peak = Ts[i_peak]
    if T_q > T_peak; return Dv_peak; end
    T_q = clamp(T_q, Ts[1], T_peak)
    for i in 1:length(Ts)-1
        if Ts[i] ≤ T_q ≤ Ts[i+1]
            frac = (T_q - Ts[i]) / (Ts[i+1] - Ts[i])
            return Ds[i] + frac * (Ds[i+1] - Ds[i])
        end
    end
    return Ds[end]
end

# ── Compound Poisson ln S ──
function cp_lnS(t, α, T, N, A, c)
    φeq = φ_eq_LSR(T)
    R2 = 1 - φeq^2; R2 ≤ 1e-10 && return 0.0
    Dv = interp_Dv(α, T)
    τ_rel = R2 / Dv
    rate_pre = A / τ_rel
    φ_min = max(0.0, φ_c_val * (φeq - sqrt(R2))) + 1e-8
    φ_zb = φ_c_val / φeq; φ_hi = min(φ_zb, 0.9999)
    logC = log_C_N(N)
    integral = 0.0
    if φ_hi > φ_min
        n_φ = 300; dφ = (φ_hi - φ_min) / n_φ
        for i in 0:n_φ
            φ = φ_min + i * dφ
            s = 1 - φ^2; s ≤ 0 && continue
            v = (φ_c_val - φeq*φ) / sqrt(s)
            fv = exp(logC + (N-3)/2*log(s))
            if v ≤ 0
                λ = rate_pre
            else
                v2R2 = v^2/R2; v2R2 ≥ 0.9999 && continue
                λ = rate_pre * exp(-c * (N-3)/2 * (-log(1 - v2R2)))
            end
            integral += (1 - exp(-λ * t)) * fv * dφ
        end
    end
    if φ_zb < 0.9999
        dφ2 = (0.9999 - φ_zb) / 50
        for i in 0:50
            φ = φ_zb + i * dφ2; s = 1-φ^2; s ≤ 0 && continue
            fv = exp(logC + (N-3)/2*log(s))
            integral += (1 - exp(-rate_pre * t)) * fv * dφ2
        end
    end
    return -(M_PAT - 1) * integral
end

# ── Load v16 data ──
struct PanelData
    α::Float64; T::Float64; N::Float64
    steps::Vector{Float64}; surv::Vector{Float64}
end

panels = PanelData[]
v16_dir = @__DIR__
for f in filter(f -> startswith(f, "v17_Pesc_a") && endswith(f, ".csv"), readdir(v16_dir))
    m_a = match(r"a(\d+\.\d+)", f); m_T = match(r"T(\d+\.\d+)", f)
    m_a === nothing && continue; m_T === nothing && continue
    α = parse(Float64, m_a.captures[1]); T = parse(Float64, m_T.captures[1])
    N = log(M_PAT) / α
    lines = readlines(joinpath(v16_dir, f))
    steps = Float64[]; pesc = Float64[]
    for line in lines[2:end]
        parts = split(line, ","); length(parts) < 2 && continue
        push!(steps, parse(Float64, parts[1]))
        push!(pesc, parse(Float64, parts[2]))
    end
    surv = max.(1.0 .- pesc, 1e-10)
    push!(panels, PanelData(α, T, N, steps, surv))
end
@printf("Loaded %d v16 panels\n", length(panels))

# ── MSE function ──
function eval_mse(logA, c; stride=3)
    A = exp(logA)
    total_err = 0.0; n_pts = 0
    for pd in panels
        idx = 1:stride:length(pd.steps)
        mask = (pd.surv[idx] .> 0.01) .& (pd.surv[idx] .< 0.99)
        sum(mask) < 3 && continue
        sel = idx[mask]
        for k in sel
            lnS_m = cp_lnS(pd.steps[k], pd.α, pd.T, pd.N, A, c)
            lnS_d = log(pd.surv[k])
            total_err += (lnS_d - lnS_m)^2
            n_pts += 1
        end
    end
    return n_pts > 0 ? total_err / n_pts : Inf, n_pts
end

# ── Fixed (A, c) — original barrier c×(N-3)/2 ──
A_best = 0.03793
best_c = 0.2816
best_logA = log(A_best)
@printf("Using A=%.4e, c=%.4f\n", A_best, best_c)

# ── τ_model vs τ_data (from Poisson fits) ──
println("\nComputing τ_model vs τ_data...")

# Extract τ_data from each panel using Poisson fit
function fit_tau_poisson(steps, surv)
    pesc = 1.0 .- surv
    t0 = 0.0
    for p_hi in [0.20, 0.35, 0.50, 0.80]
        mask = (pesc .> 0.005) .& (pesc .< p_hi)
        sum(mask) < 4 && continue
        t = steps[mask]; y = -log.(1.0 .- pesc[mask])
        X = hcat(ones(length(t)), t)
        β = X \ y
        β[2] ≤ 0 && continue
        t0 = max(0.0, -β[1]/β[2])
        break
    end
    mask_fit = (surv .> 0.005) .& (steps .> t0)
    sum(mask_fit) < 5 && return NaN
    t_m = steps[mask_fit]; lnS = log.(surv[mask_fit])
    tau_guess = length(steps) > 0 ? steps[end] / 5 : 1000.0
    best_K = NaN; best_tch = NaN; best_err = Inf
    for log_r in range(-2, 2, length=400)
        tch = tau_guess * 10^log_r
        u = @. 1 - exp(-(t_m - t0) / tch)
        K = -dot(u, lnS) / dot(u, u)
        K ≤ 0 && continue
        err = sum(@. (lnS + K*u)^2)
        if err < best_err
            best_err = err; best_K = K; best_tch = tch
        end
    end
    return isnan(best_K) ? NaN : best_tch / best_K
end

τ_data_arr = Float64[]; τ_model_arr = Float64[]
α_arr = Float64[]; T_arr = Float64[]
for pd in panels
    τ_d = fit_tau_poisson(pd.steps, pd.surv)
    isnan(τ_d) && continue
    # τ_model from compound Poisson initial slope
    dt = pd.steps[2] - pd.steps[1]
    lnS0 = cp_lnS(0.0, pd.α, pd.T, pd.N, A_best, best_c)
    lnS1 = cp_lnS(dt, pd.α, pd.T, pd.N, A_best, best_c)
    λ_model = -(lnS1 - lnS0) / dt
    τ_m = λ_model > 0 ? 1.0/λ_model : Inf
    push!(τ_data_arr, τ_d); push!(τ_model_arr, τ_m)
    push!(α_arr, pd.α); push!(T_arr, pd.T)
    @printf("  α=%.2f T=%.2f  τ_data=%.0f  τ_model=%.0f\n", pd.α, pd.T, τ_d, τ_m)
end

# ── PLOT ──
println("Plotting...")
FIG_W = round(Int, 86 / 25.4 * 100)
FIG_H = round(Int, 86 / 25.4 * 100)

tick_vals = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
tick_labs = ["10²", "10³", "10⁴", "10⁵", "10⁶", "10⁷"]
p2 = scatter(τ_data_arr, τ_model_arr,
    xscale=:log10, yscale=:log10,
    xlabel=L"\tau_\mathrm{data}", ylabel=L"\tau_\mathrm{model}",
    title=L"\tau" * " comparison",
    titlefontsize=8, framestyle=:box,
    xticks=(tick_vals, tick_labs), yticks=(tick_vals, tick_labs),
    xlims=(1e2, 1e7), ylims=(1e2, 1e7),
    aspect_ratio=:equal,
    markersize=5, markerstrokewidth=0.5,
    zcolor=α_arr, color=cgrad(:rainbow),
    colorbar=false,
    clims=(0.17, 0.23),
    label=false)
# y=x line
τ_range = [minimum(τ_data_arr)/2, maximum(τ_data_arr)*2]
plot!(p2, τ_range, τ_range, color=:black, lw=1, ls=:dash, label="y=x", legend=:topleft)
# y=10x and y=x/10 bands
plot!(p2, τ_range, 10 .* τ_range, color=:gray, lw=0.5, ls=:dot, label=false)
plot!(p2, τ_range, τ_range ./ 10, color=:gray, lw=0.5, ls=:dot, label=false)
# T_MC vertical line
T_MC = 2^15 + 2^13  # 40960
vline!(p2, [T_MC], color=:red, lw=1.5, ls=:solid, label=L"T_\mathrm{MC}")

# Manual colorbar as a narrow heatmap subplot
cb_vals = range(0.17, 0.23, length=100)
cb_ticks = [0.18, 0.20, 0.22]
cb_data = reshape(cb_vals, :, 1)
p_cb = heatmap([0], cb_vals, cb_data,
    color=cgrad(:rainbow), clims=(0.17, 0.23),
    yticks=(cb_ticks, [@sprintf("%.2f", t) for t in cb_ticks]),
    xticks=false, xlabel="", ylabel=L"\alpha",
    colorbar=false, framestyle=:box,
    tickfontsize=6, guidefontsize=7,
    left_margin=0Plots.mm, right_margin=0Plots.mm)

fig = plot(p2, p_cb, layout=@layout([a{0.92w} b{0.08w}]),
           size=(FIG_W + 50, FIG_H), dpi=300,
           left_margin=3Plots.mm, bottom_margin=3Plots.mm)

for ext in ("png", "pdf")
    savefig(fig, joinpath(out_dir, "Ac_calibration.$ext"))
end
println("Saved: panels_paper/Ac_calibration.{png,pdf}")
