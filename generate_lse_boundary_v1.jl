#=
GPU-Accelerated LSE Boundary Characterization (v5)
- Same observables as LSR v5 for direct comparison
- Identical structure: per-α processing, dense T sampling, heating protocol
- Output: lse_boundary_v5.csv [α, T, E_mean, E_std, φ_mean, φ_std, φ_min, φ_max, τ_esc, accept_rate]
- Demonstrates: why LSE has no blue bay (soft tails, no barriers)
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

const F = Float32

# ──────────────── Parameters ────────────────
const betanet = F(1.0)  # LSE kernel parameter

# Same α focus as LSR for comparison
const alpha_vec = F[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
const T_vec = F.(10 .^ range(-1.3, log10(2.5), length=40))  # 40 log-spaced points

const n_alpha = length(alpha_vec)
const n_T = length(T_vec)

const N_TRIALS = 256
const N_SAMP = 500
const MIN_PAT = 200  # LSE can use fewer patterns
const MAX_PAT = 5000

const N_EQ_INIT = 50000
const N_EQ_STEP = 10000
const ESCAPE_THRESHOLD = F(0.7)

# ──────────────── Adaptive functions ────────────────
function n_patterns(alpha)
    slope = (MAX_PAT - MIN_PAT) / (alpha_vec[end] - alpha_vec[1])
    return MIN_PAT + slope * (alpha - alpha_vec[1])
end

adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── LSE Energy ────────────────
function compute_energy_lse_batched!(E::CuVector{F}, x::CuArray{F,3},
                                      patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                      Nf::F)
    # x: [N, 1, N_TRIALS], patterns: [N, P, N_TRIALS], overlap: [P, 1, N_TRIALS]
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    
    # LSE: E = -1/β * log Σ_p exp(-β/2 * ||x - ξ_p||²)
    # = -1/β * log Σ_p exp(β/2 * (N - overlap_p))
    @. overlap = -betanet * (Nf - overlap) / 2  # Log-arguments (scaled)
    
    # Log-sum-exp trick (prevent overflow)
    m = maximum(overlap, dims=1)  # [1, 1, N_TRIALS]
    @. overlap = exp(overlap - m)
    s = sum(overlap, dims=1)  # [1, 1, N_TRIALS]
    
    E .= vec(@. -2 * (m + log(s)))  # Return to energy scale
    return nothing
end

# ──────────────── MC Step for single T ────────────────
function mc_step_single_T!(x::CuArray{F,3}, xp::CuArray{F,3},
                           E::CuVector{F}, Ep::CuVector{F},
                           pat::CuArray{F,3}, ov::CuArray{F,3},
                           β::F, ra::CuVector{F},
                           Nf::F, ss::F)
    # Propose on sphere
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) .* xp ./ nrm

    # Proposed energy
    compute_energy_lse_batched!(Ep, xp, pat, ov, Nf)

    # Metropolis accept/reject
    CUDA.rand!(ra)
    acc = @. ra < exp(-β * (Ep - E))
    
    @. x = ifelse(acc, xp, x)
    @. E = ifelse(acc, Ep, E)
    
    return sum(acc) / size(x, 3)
end

# ──────────────── Escape time test ────────────────
function measure_escape_time!(x::CuArray{F,3}, xp::CuArray{F,3},
                               E::CuVector{F}, Ep::CuVector{F},
                               tgt::CuArray{F,3}, pat::CuArray{F,3}, ov::CuArray{F,3},
                               β::F, ra::CuVector{F}, Nf::F, ss::F,
                               max_steps::Int = 10000)::Int
    phi_running = vec(mean(tgt .* x, dims=1)) ./ Nf
    
    for step in 1:max_steps
        mc_step_single_T!(x, xp, E, Ep, pat, ov, β, ra, Nf, ss)
        phi_running = vec(mean(tgt .* x, dims=1)) ./ Nf
        
        if mean(phi_running) < ESCAPE_THRESHOLD
            return step
        end
    end
    
    return max_steps
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 80)
    println("LSE Boundary Characterization v5 (Comparison with LSR)")
    println("=" ^ 80)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # Compute N(α), P(α)
    Ns = Int[]; Ps = Int[]
    for α in alpha_vec
        Pt = n_patterns(α)
        N = max(round(Int, log(Pt) / α), 2)
        push!(Ns, N); push!(Ps, round(Int, Pt))
    end
    @printf("Focus α values: %d (same as LSR v5 for direct comparison)\n", n_alpha)
    @printf("N range: %d – %d,  P range: %d – %d (fewer patterns than LSR)\n", extrema(Ns)..., extrema(Ps)...)
    @printf("T grid: %d log-spaced points, identical to LSR v5\n", n_T)
    @printf("Trials: %d,  Sampling: %d steps/T,  Heating: %d init + %d per step\n",
            N_TRIALS, N_SAMP, N_EQ_INIT, N_EQ_STEP)
    println()

    csv_file = "lse_boundary_v5.csv"
    csv_h = open(csv_file, "w")
    write(csv_h, "alpha,T,E_mean,E_std,phi_mean,phi_std,phi_min,phi_max,tau_esc,accept_rate\n")
    flush(csv_h)

    println("Processing α values...")
    
    for (i, α) in enumerate(alpha_vec)
        N = Ns[i]; P = Ps[i]; Nf = F(N)
        
        @printf("\n[%d/%d] α=%.2f, N=%d, P=%d\n", i, n_alpha, α, N, P)

        Random.seed!(42 + i)
        p_cpu = randn(F, N, P, N_TRIALS)
        for t in 1:N_TRIALS, j in 1:P
            c = @view p_cpu[:, j, t]
            c .*= sqrt(Nf) / norm(c)
        end
        pat_g = CuArray(p_cpu)
        tgt_g = CuArray(p_cpu[:, 1:1, :])

        x_g = CUDA.zeros(F, N, 1, N_TRIALS)
        xp_g = CUDA.zeros(F, N, 1, N_TRIALS)
        ov_g = CUDA.zeros(F, P, 1, N_TRIALS)
        E_g = CUDA.zeros(F, N_TRIALS)
        Ep_g = CUDA.zeros(F, N_TRIALS)
        ra_g = CUDA.zeros(F, N_TRIALS)
        ss = adaptive_ss(N)

        E_acc = CUDA.zeros(F, N_TRIALS)
        phi_acc = CUDA.zeros(F, N_TRIALS)
        accept_acc = CUDA.zeros(F, 1)
        
        T_cold = T_vec[1]
        β_cold = F(1.0) / T_cold
        
        @printf("  Initializing and heavy equilibration at T=%.4f...\n", T_cold)
        x_g .= tgt_g .+ F(0.05) .* CUDA.randn(F, N, 1, N_TRIALS)
        nrm = sqrt.(sum(x_g .^ 2, dims=1))
        x_g .= sqrt(Nf) .* x_g ./ nrm
        compute_energy_lse_batched!(E_g, x_g, pat_g, ov_g, Nf)

        prog = Progress(N_EQ_INIT, desc="  Init EQ: ")
        for step in 1:N_EQ_INIT
            mc_step_single_T!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β_cold, ra_g, Nf, ss)
            next!(prog)
        end
        finish!(prog)
        CUDA.synchronize()

        prog = Progress(n_T, desc="  T sweep : ")
        for j in 1:n_T
            T = T_vec[j]
            β = F(1.0) / T

            n_eq = (j == 1) ? 0 : N_EQ_STEP
            for step in 1:n_eq
                mc_step_single_T!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, ra_g, Nf, ss)
            end

            E_acc .= zero(F)
            phi_acc .= zero(F)
            accept_acc .= zero(F)

            for step in 1:N_SAMP
                a_rate = mc_step_single_T!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, ra_g, Nf, ss)
                E_acc .+= E_g
                phi_acc .+= vec(sum(tgt_g .* x_g, dims=1)) ./ Nf
                accept_acc[1] += a_rate
            end
            CUDA.synchronize()

            x_esc = copy(x_g)
            E_esc = copy(E_g)
            tau_esc = measure_escape_time!(x_esc, xp_g, E_esc, Ep_g, tgt_g, pat_g, ov_g, β, ra_g, Nf, ss)

            E_mean = mean(Array(E_acc)) / N_SAMP
            E_std = std(Array(E_acc)) / N_SAMP
            phi_data = Array(phi_acc) ./ N_SAMP
            phi_mean = mean(phi_data)
            phi_std = std(phi_data)
            phi_min = minimum(phi_data)
            phi_max = maximum(phi_data)
            accept_rate = Float32(accept_acc[1]) / N_SAMP

            @printf(csv_h, "%.2f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.4f\n",
                    α, T, E_mean, E_std, phi_mean, phi_std, phi_min, phi_max, tau_esc, accept_rate)
            flush(csv_h)

            next!(prog)
        end
        finish!(prog)

        CUDA.unsafe_free!(pat_g); CUDA.unsafe_free!(tgt_g)
        CUDA.unsafe_free!(x_g); CUDA.unsafe_free!(xp_g)
        CUDA.unsafe_free!(ov_g); CUDA.unsafe_free!(E_g); CUDA.unsafe_free!(Ep_g)
        CUDA.unsafe_free!(ra_g); CUDA.unsafe_free!(E_acc); CUDA.unsafe_free!(phi_acc)
        GC.gc()

        @printf("  ✓ α=%.2f complete\n", α)
    end

    close(csv_h)
    println("\n" * "=" ^ 80)
    @printf("CSV saved: %s\n", csv_file)
    @printf("Ready for LSE vs LSR comparison analysis\n")
    println("=" ^ 80)
end

main()
