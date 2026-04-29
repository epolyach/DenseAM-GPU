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

const ALPHA_VALUES = [0.18, 0.20, 0.22, 0.24, 0.26, 0.28]
const T_VALUES     = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.80]
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

# ──────────────── Compute v-component (CPU, from GPU arrays) ────────────────
# v = projection of x/√N onto e₂, where e₂ is the orthonormalized direction
# toward the most-overlapping pattern ξ^μ_max in the (ξ¹, ξ^μ) plane.
#
# e₁ = ξ¹/√N
# e₂ = (ξ^μ/√N − φ_{1μ}·e₁) / √(1−φ_{1μ}²)
# v = (x/√N)·e₂ = (φ_μ − φ₁·φ_{1μ}) / √(1−φ_{1μ}²)
#
# where φ₁ = x·ξ¹/N and φ_μ = x·ξ^μ/N.

function compute_phi_and_v(x_cpu::Array{Float32,3}, targets_cpu::Array{Float32,3},
                            mumax_cpu::Array{Float32,3}, phi_1mu::Vector{Float32},
                            N::Int)
    n_rep = size(x_cpu, 2)
    n_dis = size(x_cpu, 3)
    phi1 = zeros(Float32, n_rep, n_dis)
    v_val = zeros(Float32, n_rep, n_dis)
    for d in 1:n_dis
        sq = sqrt(1 - phi_1mu[d]^2)
        sq < 1e-6 && continue
        for r in 1:n_rep
            p1 = Float32(0)
            pm = Float32(0)
            for i in 1:N
                p1 += targets_cpu[i, 1, d] * x_cpu[i, r, d]
                pm += mumax_cpu[i, 1, d] * x_cpu[i, r, d]
            end
            p1 /= N; pm /= N
            phi1[r, d] = p1
            v_val[r, d] = (pm - p1 * phi_1mu[d]) / sq
        end
    end
    return phi1, v_val
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
    n_rep = 1  # single replica (no need for two — just measuring D)
    n_chains = n_rep * n_dis

    @printf("  α=%.2f, T=%.2f, M=%d, N=%d, n_dis=%d\n", α, T, M, N, n_dis)

    # Generate patterns on CPU
    Random.seed!(hash((α, T, M, :v13)))
    pat_cpu = randn(F, N, M, n_dis)
    for d in 1:n_dis, j in 1:M
        c = @view pat_cpu[:, j, d]
        c .*= sqrt(Nf) / norm(c)
    end
    tgt_cpu = reshape(pat_cpu[:, 1, :], N, 1, n_dis)

    # Find μ_max (most overlapping pattern) for each disorder sample
    phi_1mu = zeros(Float32, n_dis)
    mu_max_idx = zeros(Int, n_dis)
    for d in 1:n_dis
        best_ov = Float32(-Inf)
        best_j = 2
        for j in 2:M
            ov = Float32(0)
            for i in 1:N
                ov += Float32(pat_cpu[i, j, d]) * Float32(tgt_cpu[i, 1, d])
            end
            ov /= N
            if ov > best_ov
                best_ov = ov; best_j = j
            end
        end
        phi_1mu[d] = best_ov
        mu_max_idx[d] = best_j
    end
    mumax_cpu = zeros(F, N, 1, n_dis)
    for d in 1:n_dis
        mumax_cpu[:, 1, d] .= pat_cpu[:, mu_max_idx[d], d]
    end

    # Initialize replicas
    x_cpu = zeros(F, N, n_rep, n_dis)
    Random.seed!(hash((α, T, M, :v13_init)))
    initialize_near!(x_cpu, tgt_cpu, N)

    # Transfer to GPU
    pat_g = CuArray(pat_cpu)
    x_g   = CuArray(x_cpu)
    xp_g  = similar(x_g)
    ov_g  = CUDA.zeros(F, M, n_rep, n_dis)
    E_g   = CUDA.zeros(F, n_chains)
    Ep_g  = CUDA.zeros(F, n_chains)
    ra_g  = CUDA.zeros(F, n_chains)

    pat_cpu = nothing; x_cpu = nothing
    GC.gc()

    compute_energy_lsr!(E_g, x_g, pat_g, ov_g, Nf)

    # Storage for time series: φ₁(t) and v(t) at every step
    phi1_ts = zeros(Float32, N_SHORT + 1, n_chains)
    v_ts    = zeros(Float32, N_SHORT + 1, n_chains)

    # Record initial state
    x_snap = Array{Float32}(Array(x_g))
    tgt_f32 = Array{Float32}(Array(reshape(CuArray(tgt_cpu), N, 1, n_dis)))
    mumax_f32 = Array{Float32}(mumax_cpu)
    p1, vv = compute_phi_and_v(x_snap, tgt_f32, mumax_f32, phi_1mu, N)
    phi1_ts[1, :] .= vec(p1)
    v_ts[1, :]    .= vec(vv)

    # MC loop — record every step
    tgt_g = CuArray(tgt_cpu)
    t_start = time()
    for step in 1:N_SHORT
        mc_step!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, Nf, σ, ra_g)
        x_snap .= Array{Float32}(Array(x_g))
        p1, vv = compute_phi_and_v(x_snap, tgt_f32, mumax_f32, phi_1mu, N)
        phi1_ts[step + 1, :] .= vec(p1)
        v_ts[step + 1, :]    .= vec(vv)
    end
    t_elapsed = time() - t_start
    @printf("  Done: %.1f s (%.2f ms/step)\n", t_elapsed, 1000*t_elapsed/N_SHORT)

    # Compute D_v and D_φ₁ from step-to-step displacements
    # D = ⟨(Δx)²⟩ / (2·Δt) with Δt = 1 MC step
    dphi1_sq = Float64[]
    dv_sq    = Float64[]
    for j in 1:n_chains
        for t in 1:N_SHORT
            push!(dphi1_sq, (phi1_ts[t+1, j] - phi1_ts[t, j])^2)
            push!(dv_sq,    (v_ts[t+1, j]    - v_ts[t, j])^2)
        end
    end

    D_phi1 = mean(dphi1_sq) / 2
    D_v    = mean(dv_sq) / 2
    D_phi1_std = std(dphi1_sq) / (2 * sqrt(length(dphi1_sq)))
    D_v_std    = std(dv_sq) / (2 * sqrt(length(dv_sq)))

    @printf("  D_v = %.3e ± %.1e,  D_φ₁ = %.3e ± %.1e,  ratio D_v/D_φ₁ = %.3f\n",
            D_v, D_v_std, D_phi1, D_phi1_std, D_v / D_phi1)
    @printf("  ⟨φ_{1,max}⟩ = %.3f\n", mean(phi_1mu))

    # Append to summary
    open(summary_file, "a") do f
        @printf(f, "%.2f,%.2f,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.4f\n",
                α, T, N, M, n_dis, D_v, D_phi1, D_v_std, D_phi1_std, mean(phi_1mu))
    end

    # Free GPU
    CUDA.unsafe_free!(pat_g); CUDA.unsafe_free!(tgt_g)
    CUDA.unsafe_free!(x_g);   CUDA.unsafe_free!(xp_g)
    CUDA.unsafe_free!(ov_g);  CUDA.unsafe_free!(E_g)
    CUDA.unsafe_free!(Ep_g);  CUDA.unsafe_free!(ra_g)
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
