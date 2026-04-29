#=
Independent measurement of diffusion coefficient D_v (v13)
────────────────────────────────────────────────────────────────────────
Usage:
  julia measure_D_v.jl
  julia measure_D_v.jl --fresh

Goal: Measure the MC diffusion coefficient in the escape direction v
  INDEPENDENTLY from the escape-time measurement. This allows a clean
  Kramers validation: predict τ from (D_v, ΔF/T) and compare with τ_Poisson.

Method:
  At each (α, T, M):
  1. Generate patterns, initialize at retrieval (φ_eq, v=0)
  2. Run N_SHORT MC steps (≪ τ_escape), recording v(t) every step
  3. Compute D_v = ⟨Δv²⟩/(2·Δt) from the FIRST steps (no escape bias)

  v is the component of x/√N along e₂ = (ξ^μ/√N − φ_{1μ}·e₁)/√(1−φ_{1μ}²),
  where μ is the pattern with the largest overlap φ_{1μ} = φ_{1,max}.

  Also measure D_φ₁ = ⟨Δφ₁²⟩/(2·Δt) for comparison.

Output:
  v13_diffusion.csv: α, T, N, M, n_dis, D_v, D_phi1, D_v_std, D_phi1_std

Design:
  - GPU batched (same as v11m)
  - Short runs: N_SHORT = 256 MC steps (well below any escape time)
  - Record EVERY step (stride=1)
  - Large n_dis for statistics (auto-sized to GPU memory)
  - Compute v by projecting x onto the (ξ¹, ξ^μ_max) plane
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

# ══════════════════════════════════════════════════════════════════════
#                         USER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

const ALPHA_VALUES = [0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
const T_VALUES     = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.80, 1.00, 1.50, 2.00]
const M_VALUES     = [20_000]

const GPU_MEM_TARGET_GB = 35.0
const N_DIS_MAX         = 4000   # more samples for diffusion measurement (cheap: short runs)
const N_SHORT           = 256    # MC steps per trial (≪ escape time)
const SUMMARY_FILE      = "v13_diffusion.csv"

# ══════════════════════════════════════════════════════════════════════

const F = Float16
const b_lsr      = F(2 + sqrt(2))
const PHI_MIN    = F(0.75)
const PHI_MAX    = F(1.0)
const INF_ENERGY = F(1e30)

function auto_n_dis(N::Int, M::Int)
    bytes_per_dis = 2 * (N * M + 2 * N * 2 + M * 2 * 2)
    return min(floor(Int, GPU_MEM_TARGET_GB * 1e9 / bytes_per_dis), N_DIS_MAX)
end

# ──────────────── LSR Energy (batched) ────────────────
function compute_energy_lsr!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F)
    Nb = Nf / b_lsr
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = max(zero(F), one(F) - b_lsr + b_lsr * overlap / Nf)
    s = sum(overlap, dims=1)
    E .= vec(@. ifelse(s > zero(F), -Nb * log(s), INF_ENERGY))
    return nothing
end

# ──────────────── MC Step (batched) ────────────────
function mc_step!(x::CuArray{F,3}, xp::CuArray{F,3},
                  E::CuVector{F}, Ep::CuVector{F},
                  pat::CuArray{F,3}, ov::CuArray{F,3},
                  β::F, Nf::F, σ::F, ra::CuVector{F})
    randn!(xp)
    @. xp = x + σ * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm
    compute_energy_lsr!(Ep, xp, pat, ov, Nf)
    rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-β * (Ep - E)))
    n_rep = size(x, 2)
    n_dis = size(x, 3)
    a3 = reshape(acc, 1, n_rep, n_dis)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Initialize near target (CPU) ────────────────
function initialize_near!(x::Array{F,3}, targets::Array{F,3}, N::Int)
    n_rep, n_dis = size(x, 2), size(x, 3)
    for d in 1:n_dis
        target = @view targets[:, 1, d]
        for r in 1:n_rep
            phi_init = PHI_MIN + (PHI_MAX - PHI_MIN) * rand(F)
            x_perp = randn(F, N)
            ov = dot(target, x_perp) / N
            x_perp .-= ov .* target
            x_perp ./= norm(x_perp)
            x[:, r, d] .= phi_init .* target .+ sqrt(1 - phi_init^2) .* sqrt(F(N)) .* x_perp
        end
    end
end

# ──────────────── GPU projection: compute φ₁ and v from x ────────────────
# v = (φ_μ − φ₁·φ_{1μ}) / √(1−φ_{1μ}²)
# where φ₁ = x·ξ¹/N and φ_μ = x·ξ^μ_max/N.
# tgt_g[N,1,n_dis], mumax_g[N,1,n_dis], x_g[N,n_rep,n_dis]

function compute_phi1_v_gpu!(phi1_g::CuVector{F}, v_g::CuVector{F},
                              x_g::CuArray{F,3}, tgt_g::CuArray{F,3},
                              mumax_g::CuArray{F,3}, phi_1mu_g::CuVector{F},
                              sq_g::CuVector{F}, Nf::F,
                              ov_tgt::CuArray{F,3}, ov_mu::CuArray{F,3})
    n_rep = size(x_g, 2)
    n_dis = size(x_g, 3)
    # φ₁ = tgt' × x / N  →  ov_tgt[1, n_rep, n_dis]
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), tgt_g, x_g, zero(F), ov_tgt)
    # φ_μ = mumax' × x / N  →  ov_mu[1, n_rep, n_dis]
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), mumax_g, x_g, zero(F), ov_mu)

    p1 = vec(ov_tgt) ./ Nf    # [n_chains]
    pm = vec(ov_mu)  ./ Nf    # [n_chains]

    # v = (pm - p1 * phi_1mu) / sqrt(1-phi_1mu^2)
    # broadcast phi_1mu and sq over replicas
    phi_1mu_bc = repeat(phi_1mu_g, inner=n_rep)
    sq_bc      = repeat(sq_g, inner=n_rep)

    phi1_g .= p1
    v_g    .= (pm .- p1 .* phi_1mu_bc) ./ sq_bc
    return nothing
end

# ──────────────── Check if done ────────────────
function already_done(summary_file, α, T, M)
    !isfile(summary_file) && return false
    for line in readlines(summary_file)[2:end]
        f = split(line, ",")
        length(f) < 4 && continue
        if isapprox(parse(Float64,f[1]), α; atol=0.001) &&
           isapprox(parse(Float64,f[2]), T; atol=0.001) &&
           parse(Int, f[4]) == M
            return true
        end
    end
    return false
end

# ──────────────── Run one probe point ────────────────
function run_point!(α, T, M, n_dis, summary_file)
    N = max(round(Int, log(M) / α), 2)
    Nf = F(N)
    β = F(1 / T)
    σ = F(2.4 * T / sqrt(Float64(N)))
    n_rep = 1
    n_chains = n_rep * n_dis

    @printf("  α=%.2f, T=%.2f, M=%d, N=%d, n_dis=%d\n", α, T, M, N, n_dis)

    # Generate patterns on CPU, normalize
    Random.seed!(hash((α, T, M, :v13)))
    pat_cpu = randn(F, N, M, n_dis)
    for d in 1:n_dis, j in 1:M
        c = @view pat_cpu[:, j, d]
        c .*= sqrt(Nf) / norm(c)
    end
    tgt_cpu = reshape(pat_cpu[:, 1, :], N, 1, n_dis)

    # Transfer patterns to GPU
    pat_g = CuArray(pat_cpu)
    tgt_g = CuArray(tgt_cpu)

    # ── Find μ_max on GPU: one batched GEMM ──
    # overlaps[M, 1, n_dis] = pat_g' × tgt_g  → φ_{1μ} for all μ, all disorder
    ov_all = CUDA.zeros(F, M, 1, n_dis)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), pat_g, tgt_g, zero(F), ov_all)
    @. ov_all = ov_all / Nf
    ov_all[1, :, :] .= F(-Inf)  # exclude target (pattern 1)

    # argmax along dim 1 for each disorder sample
    ov_2d = dropdims(ov_all, dims=2)  # [M, n_dis]
    mu_max_vals, mu_max_idxs = findmax(Array(ov_2d), dims=1)  # CPU for indexing
    phi_1mu_cpu = Float32.(vec(mu_max_vals))
    mu_idx_cpu = [ci[1] for ci in vec(CartesianIndices(size(ov_2d))[mu_max_idxs])]

    CUDA.unsafe_free!(ov_all)

    # Build mumax_g[N, 1, n_dis] on CPU then transfer
    mumax_cpu = zeros(F, N, 1, n_dis)
    for d in 1:n_dis
        mumax_cpu[:, 1, d] .= pat_cpu[:, mu_idx_cpu[d], d]
    end
    mumax_g = CuArray(mumax_cpu)

    # GPU vectors for φ_{1μ} and √(1−φ_{1μ}²)
    phi_1mu_g = CuArray(F.(phi_1mu_cpu))
    sq_g      = CuArray(F.(sqrt.(max.(0, 1.0f0 .- phi_1mu_cpu.^2))))

    @printf("  ⟨φ_{1,max}⟩ = %.3f (GPU μ_max search done)\n", mean(phi_1mu_cpu))

    # Initialize replicas on CPU, transfer to GPU
    x_cpu = zeros(F, N, n_rep, n_dis)
    Random.seed!(hash((α, T, M, :v13_init)))
    initialize_near!(x_cpu, tgt_cpu, N)
    x_g  = CuArray(x_cpu)
    xp_g = similar(x_g)

    pat_cpu = nothing; x_cpu = nothing; mumax_cpu = nothing; tgt_cpu = nothing
    GC.gc()

    # MC work arrays
    ov_g  = CUDA.zeros(F, M, n_rep, n_dis)
    E_g   = CUDA.zeros(F, n_chains)
    Ep_g  = CUDA.zeros(F, n_chains)
    ra_g  = CUDA.zeros(F, n_chains)

    # Projection work arrays (small: [1, n_rep, n_dis])
    ov_tgt = CUDA.zeros(F, 1, n_rep, n_dis)
    ov_mu  = CUDA.zeros(F, 1, n_rep, n_dis)

    # GPU accumulators for ⟨Δφ₁²⟩ and ⟨Δv²⟩
    phi1_prev = CUDA.zeros(F, n_chains)
    v_prev    = CUDA.zeros(F, n_chains)
    phi1_cur  = CUDA.zeros(F, n_chains)
    v_cur     = CUDA.zeros(F, n_chains)
    sum_dphi1_sq = CUDA.zeros(Float32, n_chains)
    sum_dv_sq    = CUDA.zeros(Float32, n_chains)

    compute_energy_lsr!(E_g, x_g, pat_g, ov_g, Nf)

    # Record initial (φ₁, v) on GPU
    compute_phi1_v_gpu!(phi1_prev, v_prev, x_g, tgt_g, mumax_g,
                         phi_1mu_g, sq_g, Nf, ov_tgt, ov_mu)

    # MC loop — accumulate Δ² on GPU, no CPU transfer
    t_start = time()
    for step in 1:N_SHORT
        mc_step!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, Nf, σ, ra_g)
        compute_phi1_v_gpu!(phi1_cur, v_cur, x_g, tgt_g, mumax_g,
                             phi_1mu_g, sq_g, Nf, ov_tgt, ov_mu)
        # Accumulate Δ² (in Float32 for stability)
        @. sum_dphi1_sq += Float32(phi1_cur - phi1_prev)^2
        @. sum_dv_sq    += Float32(v_cur - v_prev)^2
        # Swap
        phi1_prev .= phi1_cur
        v_prev    .= v_cur
    end
    CUDA.synchronize()
    t_elapsed = time() - t_start
    @printf("  Done: %.1f s (%.2f ms/step)\n", t_elapsed, 1000*t_elapsed/N_SHORT)

    # Fetch and compute D
    dphi1_sq_cpu = Array(sum_dphi1_sq)
    dv_sq_cpu    = Array(sum_dv_sq)

    # D = mean(Σ Δx²) / (2 · N_SHORT)  per chain, then average over chains
    D_phi1_per = dphi1_sq_cpu ./ (2 * N_SHORT)
    D_v_per    = dv_sq_cpu    ./ (2 * N_SHORT)

    D_phi1 = mean(D_phi1_per)
    D_v    = mean(D_v_per)
    D_phi1_std = std(D_phi1_per) / sqrt(n_chains)
    D_v_std    = std(D_v_per) / sqrt(n_chains)

    @printf("  D_v = %.3e ± %.1e,  D_φ₁ = %.3e ± %.1e,  ratio D_v/D_φ₁ = %.3f\n",
            D_v, D_v_std, D_phi1, D_phi1_std, D_v / max(D_phi1, 1e-30))
    @printf("  ⟨φ_{1,max}⟩ = %.3f\n", mean(phi_1mu_cpu))

    # Append to summary
    open(summary_file, "a") do f
        @printf(f, "%.2f,%.2f,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.4f\n",
                α, T, N, M, n_dis, D_v, D_phi1, D_v_std, D_phi1_std, mean(phi_1mu_cpu))
    end

    # Free GPU
    for arr in [pat_g, tgt_g, mumax_g, x_g, xp_g, ov_g, E_g, Ep_g, ra_g,
                ov_tgt, ov_mu, phi_1mu_g, sq_g,
                phi1_prev, v_prev, phi1_cur, v_cur, sum_dphi1_sq, sum_dv_sq]
        CUDA.unsafe_free!(arr)
    end
    GC.gc(); CUDA.reclaim()
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")
    fresh = "--fresh" in ARGS

    probes = Tuple{Float64,Float64,Int}[]
    for α in ALPHA_VALUES, T in T_VALUES, M in M_VALUES
        push!(probes, (α, T, M))
    end

    println("=" ^ 70)
    println("Diffusion Coefficient Measurement – GPU v13")
    println("  α: $ALPHA_VALUES")
    println("  T: $T_VALUES")
    println("  M: $M_VALUES")
    println("  Points: $(length(probes))")
    println("  MC steps per point: $N_SHORT (stride=1)")
    println("=" ^ 70)

    if fresh || !isfile(SUMMARY_FILE)
        open(SUMMARY_FILE, "w") do f
            write(f, "alpha,T,N,M,n_dis,D_v,D_phi1,D_v_std,D_phi1_std,phi_1max_mean\n")
        end
    end

    n_done = 0; n_skip = 0
    for (pi, (α, T, M)) in enumerate(probes)
        N = max(round(Int, log(M) / α), 2)
        n_dis = auto_n_dis(N, M)

        if !fresh && already_done(SUMMARY_FILE, α, T, M)
            n_skip += 1
            @printf("── Point %d/%d: α=%.2f, T=%.2f, M=%d — SKIP ──\n",
                    pi, length(probes), α, T, M)
            continue
        end

        @printf("\n── Point %d/%d ──\n", pi, length(probes))
        run_point!(α, T, M, n_dis, SUMMARY_FILE)
        n_done += 1
    end

    println("\n" * "=" ^ 70)
    @printf("Complete. %d run, %d skipped. Output: %s\n", n_done, n_skip, SUMMARY_FILE)
    println("=" ^ 70)
end

main()
