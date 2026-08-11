#=
Global (A, c) fit of compound Poisson survival model to all v14 data
────────────────────────────────────────────────────────────────────────
Model:
  S(t) = exp(-(M-1) ∫ (1 - exp(-λ(φ)t)) f_exact(φ,N) dφ )
  λ(φ) = (A / τ_rel) exp(-c ΔF(φ)/T)
  τ_rel = R² / D_v(α,T)   [from v13 measurements]

Scan (A, c), compute total MSE of ln S across all panels.
────────────────────────────────────────────────────────────────────────
=#

using Printf, LinearAlgebra, SpecialFunctions, Statistics

const M_PAT = 20000
const b_lsr = 2 + sqrt(2)
const φ_c   = (b_lsr - 1) / b_lsr

function φ_eq_LSR(T)
    T < 1e-10 && return 1.0
    φ = 0.95
    for _ in 1:200
        D = 1 - b_lsr + b_lsr*φ
        D ≤ 1e-10 && (φ = φ_c + 0.005; continue)
        f = (1 - φ^2) - T*φ*D
        fp = -2φ - T*(D + b_lsr*φ)
        φ = clamp(φ - f/fp, φ_c + 1e-8, 1 - 1e-8)
    end
    return φ
end

function log_C_N(N)
    loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)
end

# ──────────────── Read v13 D_v ────────────────
v13_lines = readlines(joinpath(@__DIR__, "v13_diffusion.csv"))
v13_α = Float64[]; v13_T = Float64[]; v13_Dv = Float64[]
for line in v13_lines[2:end]
    f = split(line, ",")
    length(f) < 6 && continue
    push!(v13_α, parse(Float64, f[1]))
    push!(v13_T, parse(Float64, f[2]))
    push!(v13_Dv, parse(Float64, f[6]))
end
v13_alphas = sort(unique(v13_α))

function interp_Dv(α_q, T_q)
    _, ia = findmin(abs.(v13_alphas .- α_q))
    α_near = v13_alphas[ia]
    mask = v13_α .== α_near
    Ts_a = v13_T[mask]; Ds_a = v13_Dv[mask]
    perm = sortperm(Ts_a); Ts_a = Ts_a[perm]; Ds_a = Ds_a[perm]
    T_q = clamp(T_q, Ts_a[1], Ts_a[end])
    for i in 1:length(Ts_a)-1
        if Ts_a[i] ≤ T_q ≤ Ts_a[i+1]
            frac = (T_q - Ts_a[i]) / (Ts_a[i+1] - Ts_a[i])
            return Ds_a[i] + frac * (Ds_a[i+1] - Ds_a[i])
        end
    end
    return Ds_a[end]
end

# ──────────────── Read all v14 files ────────────────
# Use v16 data (confirmed barrier crossing — clean Kramers escape)
v14_dir = @__DIR__
all_files = filter(f -> startswith(f, "v16_Pesc_a") && endswith(f, ".csv"), readdir(v14_dir))

struct PanelData
    α::Float64; T::Float64; N::Int
    steps::Vector{Float64}; surv::Vector{Float64}
end

panels = PanelData[]
for f in all_files
    m_a = match(r"a(\d+\.\d+)", f); m_T = match(r"T(\d+\.\d+)", f)
    m_a === nothing && continue; m_T === nothing && continue
    α = parse(Float64, m_a.captures[1]); T = parse(Float64, m_T.captures[1])
    N = max(round(Int, log(M_PAT)/α), 2)
    lines = readlines(joinpath(v14_dir, f))
    steps = Float64[]; pesc = Float64[]
    for line in lines[2:end]
        parts = split(line, ","); length(parts) < 2 && continue
        push!(steps, parse(Float64, parts[1]))
        push!(pesc, parse(Float64, parts[2]))
    end
    surv = max.(1.0 .- pesc, 1e-10)
    push!(panels, PanelData(α, T, N, steps, surv))
end
@printf("Loaded %d panels\n", length(panels))

# ──────────────── Compound Poisson S(t) ────────────────
# S(t) = exp(-(M-1) ∫ (1-exp(-λ(φ)t)) f(φ,N) dφ)
# λ(φ) = (A/τ_rel) exp(-c ΔF(φ)/T)

function compound_poisson_lnS(t_arr, α, T, N, A, c)
    φeq = φ_eq_LSR(T)
    R2 = 1 - φeq^2
    R2 ≤ 1e-10 && return zeros(length(t_arr))

    Dv = interp_Dv(α, T)
    τ_rel = R2 / Dv
    rate_prefactor = A / τ_rel

    φ_min = max(0.0, φ_c * (φeq - sqrt(R2))) + 1e-8
    φ_zb = φ_c / φeq
    φ_hi = min(φ_zb, 0.9999)

    logC = log_C_N(N)
    n_φ = 500
    if φ_hi ≤ φ_min
        return zeros(length(t_arr))
    end
    dφ = (φ_hi - φ_min) / n_φ

    # Precompute per-channel rates λ(φ_i)
    φ_pts = [φ_min + i * dφ for i in 0:n_φ]
    λ_pts = Float64[]
    f_pts = Float64[]
    for φ in φ_pts
        s = 1 - φ^2; s ≤ 0 && (push!(λ_pts, 0.0); push!(f_pts, 0.0); continue)
        v = (φ_c - φeq * φ) / sqrt(s)
        v ≤ 0 && (push!(λ_pts, rate_prefactor); push!(f_pts, exp(logC + (N-3)/2*log(s))); continue)
        v2_R2 = v^2 / R2
        v2_R2 ≥ 0.9999 && (push!(λ_pts, 0.0); push!(f_pts, 0.0); continue)
        ΔF_T = (N-3)/2 * (-log(1 - v2_R2))
        λ = rate_prefactor * exp(-c * ΔF_T)
        fv = exp(logC + (N-3)/2 * log(s))
        push!(λ_pts, λ)
        push!(f_pts, fv)
    end

    # Barrierless region (φ > φ_zb)
    if φ_zb < 0.9999
        dφ2 = (0.9999 - φ_zb) / 100
        for i in 0:100
            φ = φ_zb + i * dφ2
            s = 1-φ^2; s ≤ 0 && continue
            push!(φ_pts, φ)
            push!(λ_pts, rate_prefactor)
            push!(f_pts, exp(logC + (N-3)/2*log(s)))
        end
    end

    # Compute ln S(t) for each t
    n_ch = length(φ_pts)
    lnS = zeros(length(t_arr))
    for (it, t) in enumerate(t_arr)
        integral = 0.0
        for j in 1:n_ch
            λj = λ_pts[j]; fj = f_pts[j]
            fj ≤ 0 && continue
            contrib = (1 - exp(-λj * t)) * fj
            # Use appropriate dφ
            d = j ≤ n_φ+1 ? dφ : (0.9999 - φ_zb)/100
            integral += contrib * d
        end
        lnS[it] = -(M_PAT - 1) * integral
    end
    return lnS
end

# ──────────────── Global fit: scan (A, c) ────────────────
function global_fit(panels)
    println("\nScanning (A, c)...")
    println("  Using D_v from v13 for τ_rel")

    best_A = 0.0; best_c = 0.0; best_mse = Inf

    function eval_mse(logA, c, panels; stride=3)
        A = exp(logA)
        total_err = 0.0; n_pts = 0
        for pd in panels
            idx = 1:stride:length(pd.steps)
            mask = (pd.surv[idx] .> 0.27) .& (pd.surv[idx] .< 0.99)
            sum(mask) < 3 && continue
            sel = idx[mask]
            lnS_model = compound_poisson_lnS(pd.steps[sel], pd.α, pd.T, pd.N, A, c)
            lnS_data = log.(pd.surv[sel])
            total_err += sum((lnS_data .- lnS_model).^2)
            n_pts += length(sel)
        end
        return n_pts > 0 ? total_err / n_pts : Inf
    end

    # Filter: only metastable points (τ_data > T_MC = 40960)
    T_MC = 2^15 + 2^13
    meta_panels = PanelData[]
    for pd in panels
        pesc_final = 1.0 - pd.surv[end]
        # Rough τ estimate: if P_esc < 0.5, likely τ > T_run > T_MC
        # More precisely: use the survival curve
        # A panel is metastable if most trials DON'T escape
        if pesc_final < 0.63  # P_esc < 1-e⁻¹ → τ > T_run
            push!(meta_panels, pd)
            @printf("  META: α=%.2f T=%.2f (P_esc=%.3f)\n", pd.α, pd.T, pesc_final)
        else
            @printf("  skip: α=%.2f T=%.2f (P_esc=%.3f)\n", pd.α, pd.T, pesc_final)
        end
    end
    @printf("  Using %d metastable panels (of %d total)\n", length(meta_panels), length(panels))
    panels = meta_panels  # use only metastable for fitting

    # Start from known good point
    best_A = 0.025; best_c = 0.25
    best_mse = eval_mse(log(best_A), best_c, panels)
    @printf("  Start: A=%.3e, c=%.3f, MSE=%.4f\n", best_A, best_c, best_mse)

    # Newton (finite-difference) optimization on (logA, c)
    logA = log(best_A); c = best_c
    ε = 1e-3  # finite difference step
    @printf("  Newton iterations:\n")
    for iter in 1:30
        f0 = eval_mse(logA, c, panels)
        # Gradient
        dfdA = (eval_mse(logA+ε, c, panels) - eval_mse(logA-ε, c, panels)) / (2ε)
        dfdc = (eval_mse(logA, c+ε, panels) - eval_mse(logA, c-ε, panels)) / (2ε)
        # Hessian
        d2fdA2 = (eval_mse(logA+ε, c, panels) - 2f0 + eval_mse(logA-ε, c, panels)) / ε^2
        d2fdc2 = (eval_mse(logA, c+ε, panels) - 2f0 + eval_mse(logA, c-ε, panels)) / ε^2
        d2fdAc = (eval_mse(logA+ε, c+ε, panels) - eval_mse(logA+ε, c-ε, panels) -
                  eval_mse(logA-ε, c+ε, panels) + eval_mse(logA-ε, c-ε, panels)) / (4ε^2)
        H = [d2fdA2 d2fdAc; d2fdAc d2fdc2]
        g = [dfdA; dfdc]
        det_H = H[1,1]*H[2,2] - H[1,2]*H[2,1]
        abs(det_H) < 1e-20 && (@printf("    iter %d: singular Hessian\n", iter); break)
        δ = H \ g
        # Damped step (trust region)
        step_size = min(1.0, 2.0 / norm(δ))
        logA_new = logA - step_size * δ[1]
        c_new = max(0.01, c - step_size * δ[2])
        f_new = eval_mse(logA_new, c_new, panels)
        @printf("    iter %d: A=%.4e c=%.4f MSE=%.6f\n", iter, exp(logA_new), c_new, f_new)
        if abs(f_new - f0) < 1e-6 * abs(f0)
            logA = logA_new; c = c_new; break
        end
        logA = logA_new; c = c_new
    end
    best_A = exp(logA); best_c = c; best_mse = eval_mse(logA, c, panels)

    @printf("\n  ═══ RESULT: A = %.4e, c = %.4f ═══\n", best_A, best_c)
    @printf("  MSE(ln S) = %.4f (RMS = %.3f)\n", best_mse, sqrt(best_mse))
    return best_A, best_c
end

const (BEST_A, BEST_C) = global_fit(panels)

# Per-panel τ = 1/⟨λ⟩
println("\n  Per-panel τ from compound Poisson:")
println("  α     T      N    τ_model")
for pd in panels
    dt = pd.steps[2] - pd.steps[1]
    lnS_0 = compound_poisson_lnS([0.0, dt], pd.α, pd.T, pd.N, BEST_A, BEST_C)
    λ_mean = -(lnS_0[2] - lnS_0[1]) / dt
    τ = λ_mean > 0 ? 1.0/λ_mean : Inf
    τ_str = τ ≥ 1000 ? @sprintf("%.0fk", τ/1000) : @sprintf("%.0f", τ)
    @printf("  %.2f  %.2f  %3d  %s\n", pd.α, pd.T, pd.N, τ_str)
end
