#=
v15 — Kramers escape detection via v = v_entry
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSR_v15_ventry.jl
  julia basin_stab_LSR_v15_ventry.jl --fresh

Detects the BARRIER CROSSING (Stage 1) directly by tracking the
perpendicular component v toward ξ^μ_max and detecting v ≥ v_entry.

Design:
  - Initialize at φ=1
  - Find μ_max per disorder sample (GPU GEMM)
  - Each step: compute v = (φ_μ − φ₁·φ_{1μ})/√(1−φ_{1μ}²) on GPU
  - Detect first crossing v ≥ v_entry(φ₁)
    where v_entry(φ₁) = (φ_c − φ₁·φ_{1μ})/√(1−φ_{1μ}²)
    (threshold depends on current φ₁, not just φ_eq)
  - Record P_esc(t), survival curve

Output:
  v15_summary.csv
  v15_Pesc_a{α}_T{T}.csv
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

const GPU_MEM_TARGET_GB = 35.0
const N_DIS_MAX         = 2000
const M_PAT             = 20000
const SUMMARY_FILE      = "v15_summary.csv"

# Probe grid
const ALPHA_VALUES = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
const T_VALUES     = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.80]

# Adaptive T_run and stride based on estimated τ
# Higher α and T → shorter τ → shorter run, finer stride
function get_run_params(alpha, T)
    # Rough estimate: τ decreases with α and T
    # Use conservative estimates from v14
    if T >= 0.80
        return (10_000, 4)
    elseif T >= 0.50
        return (30_000, 8)
    elseif T >= 0.40
        return (100_000, 32)
    elseif T >= 0.30
        if alpha >= 0.26; return (100_000, 64)
        else;             return (300_000, 128)
        end
    elseif T >= 0.25
        if alpha >= 0.26; return (200_000, 128)
        else;             return (500_000, 256)
        end
    elseif T >= 0.20
        if alpha >= 0.26; return (300_000, 128)
        else;             return (500_000, 256)
        end
    else  # T=0.15
        if alpha >= 0.28; return (300_000, 128)
        else;             return (500_000, 256)
        end
    end
end

const F = Float16
const b_lsr      = F(2 + sqrt(2))
const PHI_C_F64  = Float64((Float64(b_lsr)-1)/Float64(b_lsr))
const INF_ENERGY = F(1e30)

function auto_n_dis(N::Int, M::Int)
    bytes_per_dis = 2 * (N*M + N*2 + M*2 + 2)  # pat + x,xp + ov + phi1,phimu
    return min(floor(Int, GPU_MEM_TARGET_GB * 1e9 / bytes_per_dis), N_DIS_MAX)
end

# ──────────────── GPU kernels ────────────────
function compute_energy_lsr!(E, x, pat, ov, Nf)
    Nb = Nf / b_lsr
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), pat, x, zero(F), ov)
    @. ov = max(zero(F), one(F) - b_lsr + b_lsr * ov / Nf)
    s = sum(ov, dims=1)
    E .= vec(@. ifelse(s > zero(F), -Nb * log(s), INF_ENERGY))
    return nothing
end

function mc_step!(x, xp, E, Ep, pat, ov, β, Nf, σ, ra)
    randn!(xp)
    @. xp = x + σ * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm
    compute_energy_lsr!(Ep, xp, pat, ov, Nf)
    rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-β * (Ep - E)))
    a3 = reshape(acc, 1, size(x,2), size(x,3))
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
end

# ──────────────── Compute φ₁, v, and detect v ≥ v_entry ────────────────
# v = (φ_μ − φ₁·φ_{1μ}) / √(1−φ_{1μ}²)
# v_entry(φ₁) = (φ_c − φ₁·φ_{1μ}) / √(1−φ_{1μ}²)
# Escape condition: v ≥ v_entry, i.e. φ_μ − φ₁·φ_{1μ} ≥ φ_c − φ₁·φ_{1μ}
# Simplifies to: φ_μ ≥ φ_c  (the spurious pattern enters support!)
#
# So detecting v ≥ v_entry is EXACTLY detecting φ_μ ≥ φ_c.
# No need to compute v explicitly — just check if ξ^μ_max is in support.

function detect_escape_gpu!(escaped, x, tgt, mumax, phi_1mu, Nf, ov_tgt, ov_mu, phi_c_F)
    n_rep = size(x, 2); n_dis = size(x, 3)
    # φ₁ = tgt' × x / N
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), tgt, x, zero(F), ov_tgt)
    # φ_μ = mumax' × x / N
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), mumax, x, zero(F), ov_mu)
    
    phi_mu = vec(ov_mu) ./ Nf
    # Escape: φ_μ ≥ φ_c AND not already escaped
    newly = @. (phi_mu >= phi_c_F) & (escaped == Int32(0))
    @. escaped = ifelse(newly, Int32(1), escaped)
    return nothing
end

# ──────────────── Check if done ────────────────
function already_done(sf, α, T)
    !isfile(sf) && return false
    for line in readlines(sf)[2:end]
        f = split(line, ",")
        length(f) < 3 && continue
        isapprox(parse(Float64,f[1]), α; atol=0.001) &&
        isapprox(parse(Float64,f[2]), T; atol=0.001) && return true
    end
    return false
end

# ──────────────── Run one point ────────────────
function run_point!(α, T, T_run, stride, n_dis)
    N = max(round(Int, log(M_PAT) / α), 2)
    M = M_PAT; Nf = F(N)
    β = F(1/T); σ = F(2.4*T/sqrt(Float64(N)))
    n_rep = 1; n_chains = n_rep * n_dis
    n_record = T_run ÷ stride
    phi_c_F = F(PHI_C_F64)

    @printf("  α=%.2f, T=%.2f, N=%d, n_dis=%d, T_run=%d, stride=%d\n",
            α, T, N, n_dis, T_run, stride)

    # Generate patterns
    Random.seed!(hash((α, T, M, :v15)))
    pat_cpu = randn(F, N, M, n_dis)
    for d in 1:n_dis, j in 1:M
        c = @view pat_cpu[:, j, d]
        c .*= sqrt(Nf) / norm(c)
    end
    tgt_cpu = reshape(pat_cpu[:, 1, :], N, 1, n_dis)

    # Transfer to GPU
    pat_g = CuArray(pat_cpu)
    tgt_g = CuArray(tgt_cpu)

    # Find μ_max on GPU
    ov_all = CUDA.zeros(F, M, 1, n_dis)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), pat_g, tgt_g, zero(F), ov_all)
    @. ov_all = ov_all / Nf
    ov_all[1, :, :] .= F(-Inf)
    ov_2d = dropdims(ov_all, dims=2)
    mu_max_vals, mu_max_idxs = findmax(Array(ov_2d), dims=1)
    phi_1mu_cpu = Float32.(vec(mu_max_vals))
    mu_idx_cpu = [ci[1] for ci in vec(CartesianIndices(size(ov_2d))[mu_max_idxs])]
    CUDA.unsafe_free!(ov_all)

    mumax_cpu = zeros(F, N, 1, n_dis)
    for d in 1:n_dis
        mumax_cpu[:, 1, d] .= pat_cpu[:, mu_idx_cpu[d], d]
    end
    mumax_g = CuArray(mumax_cpu)

    @printf("  ⟨φ_{1,max}⟩ = %.3f, v_entry criterion: φ_μ ≥ φ_c = %.4f\n",
            mean(phi_1mu_cpu), PHI_C_F64)

    # Initialize at φ=1
    x_cpu = zeros(F, N, n_rep, n_dis)
    for d in 1:n_dis
        x_cpu[:, 1, d] .= tgt_cpu[:, 1, d]
    end
    x_g = CuArray(x_cpu); xp_g = similar(x_g)

    pat_cpu = nothing; x_cpu = nothing; tgt_cpu = nothing; mumax_cpu = nothing
    GC.gc()

    ov_g  = CUDA.zeros(F, M, n_rep, n_dis)
    E_g   = CUDA.zeros(F, n_chains)
    Ep_g  = CUDA.zeros(F, n_chains)
    ra_g  = CUDA.zeros(F, n_chains)
    ov_tgt = CUDA.zeros(F, 1, n_rep, n_dis)
    ov_mu  = CUDA.zeros(F, 1, n_rep, n_dis)

    escaped = CUDA.zeros(Int32, n_chains)
    escape_count = zeros(Int, n_record)

    compute_energy_lsr!(E_g, x_g, pat_g, ov_g, Nf)

    t_start = time()
    rec_idx = 0
    prog = Progress(T_run, desc="  MC: ", showspeed=true)
    for step in 1:T_run
        mc_step!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, Nf, σ, ra_g)

        if step % stride == 0
            rec_idx += 1
            detect_escape_gpu!(escaped, x_g, tgt_g, mumax_g,
                               nothing, Nf, ov_tgt, ov_mu, phi_c_F)
            n_esc = sum(Array(escaped))
            escape_count[rec_idx] = n_esc
            ProgressMeter.update!(prog, step,
                showvalues=[(:P_esc, @sprintf("%.3f", n_esc/n_chains))])
        else
            next!(prog)
        end
    end
    CUDA.synchronize()
    t_elapsed = time() - t_start
    @printf("  Done: %.1f s (%.3f ms/step)\n", t_elapsed, 1000*t_elapsed/T_run)

    t_arr = [(i*stride) for i in 1:n_record]
    P_arr = escape_count ./ n_chains
    P_final = P_arr[end]
    @printf("  P_esc_final = %.4f\n", P_final)

    # Save
    pesc_file = @sprintf("v15_Pesc_a%.2f_T%.2f.csv", α, T)
    open(pesc_file, "w") do f
        write(f, "step,P_esc\n")
        for i in 1:n_record
            @printf(f, "%d,%.6f\n", t_arr[i], P_arr[i])
        end
    end
    @printf("  Saved: %s\n", pesc_file)

    open(SUMMARY_FILE, "a") do f
        @printf(f, "%.2f,%.2f,%d,%d,%d,%.6f,%d,%d\n",
                α, T, N, M, n_dis, P_final, T_run, stride)
    end

    for arr in [pat_g, tgt_g, mumax_g, x_g, xp_g, ov_g, E_g, Ep_g, ra_g,
                ov_tgt, ov_mu, escaped]
        CUDA.unsafe_free!(arr)
    end
    GC.gc(); CUDA.reclaim()
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")
    fresh = "--fresh" in ARGS

    # Build probe list sorted fast-first
    probes = Tuple{Float64,Float64,Int,Int}[]
    for α in ALPHA_VALUES, T in T_VALUES
        T_run, stride = get_run_params(α, T)
        push!(probes, (α, T, T_run, stride))
    end
    sort!(probes, by=x->x[3])  # shortest T_run first

    println("=" ^ 70)
    println("v15 — Kramers Escape via v_entry (φ_μ ≥ φ_c)")
    println("  Points: $(length(probes))")
    println("=" ^ 70)

    if fresh || !isfile(SUMMARY_FILE)
        open(SUMMARY_FILE, "w") do f
            write(f, "alpha,T,N,M,n_dis,P_esc_final,T_run,stride\n")
        end
    end

    for (pi, (α, T, T_run, stride)) in enumerate(probes)
        if !fresh && already_done(SUMMARY_FILE, α, T)
            @printf("── Point %d/%d: α=%.2f, T=%.2f — SKIP ──\n", pi, length(probes), α, T)
            continue
        end
        N = max(round(Int, log(M_PAT)/α), 2)
        n_dis = auto_n_dis(N, M_PAT)
        @printf("\n── Point %d/%d ──\n", pi, length(probes))
        run_point!(α, T, T_run, stride, n_dis)
    end

    println("\n" * "=" ^ 70)
    println("Complete. Summary: $SUMMARY_FILE")
    println("=" ^ 70)
end

main()
