#=
GPU-Accelerated LSR Boundary Characterization (v5)
- Direct study of LOCAL MINIMUM DESTRUCTION via thermal effects
- Per-α processing with dense T sampling near T_c
- Four observables: free energy, escape time, order statistics, acceptance rate
- Heating protocol within each α (state propagates T → T+ΔT)
- Output: lsr_boundary_v5.csv [α, T, E_mean, E_std, φ_mean, φ_std, φ_min, φ_max, τ_esc, accept_rate]
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

const F = Float32

# ──────────────── Parameters ────────────────
const b_lsr = F(2 + sqrt(2))  # ≈ 3.414

# Focus on key α values spanning retrieval → spin-glass boundary
const alpha_vec = F[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

# Dense T grid: log-spaced from 0.05 → 2.5, with extra points near predicted T_c
# This allows precise characterization of minimum collapse
const T_vec = F.(10 .^ range(-1.3, log10(2.5), length=40))  # 40 log-spaced points

const n_alpha = length(alpha_vec)
const n_T = length(T_vec)

# Sampling parameters
const N_TRIALS = 256
const N_SAMP = 500  # Per T: balance speed vs. statistics for boundary study
const MIN_PAT = 500
const MAX_PAT = 20000

# Heating equilibration: propagate state from T → T+ΔT
const N_EQ_INIT = 50000   # Heavy equilibration at T_1 (coldest)
const N_EQ_STEP = 10000   # Re-equilibration between T steps
const ESCAPE_THRESHOLD = F(0.7)  # φ < 0.7 = escaped retrieval basin

# ──────────────── Adaptive functions ────────────────
function n_patterns(alpha)
    slope = (MAX_PAT - MIN_PAT) / (alpha_vec[end] - alpha_vec[1])
    return MIN_PAT + slope * (alpha - alpha_vec[1])
end

adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── LSR Energy ────────────────
const INF_ENERGY = F(1e30)

function compute_energy_lsr_batched!(E::CuVector{F}, x::CuArray{F,3},
                                      patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                      Nf::F)
    # x: [N, 1, N_TRIALS], patterns: [N, P, N_TRIALS], overlap: [P, 1, N_TRIALS]
    Nb = Nf / b_lsr
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = max(zero(F), one(F) - b_lsr + b_lsr * overlap / Nf)
    s = sum(overlap, dims=1)  # [1, 1, N_TRIALS]
    E .= vec(@. ifelse(s > zero(F), -Nb * log(s), INF_ENERGY))
    return nothing
end

# ──────────────── MC Step for single T ────────────────
function mc_step_single_T!(x::CuArray{F,3}, xp::CuArray{F,3},
                           E::CuVector{F}, Ep::CuVector{F},
                           pat::CuArray{F,3}, ov::CuArray{F,3},
                           β::F, ra::CuVector{F},
                           Nf::F, ss::F)
    nTrials = size(x, 3)

    # Propose on sphere
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) .* xp ./ nrm

    # Proposed energy
    compute_energy_lsr_batched!(Ep, xp, pat, ov, Nf)

    # Metropolis accept/reject (unconditionally reject if Ep is infinite — basin escape forbidden)
    CUDA.rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-β * (Ep - E)))

    # Update: accept or stay
    @. x = ifelse(acc, xp, x)
    @. E = ifelse(acc, Ep, E)
    
    return sum(acc) / nTrials  # Return acceptance rate for this step
end

# ──────────────── Escape time test ────────────────
# Measure how long state remains trapped in retrieval basin (φ > threshold)
function measure_escape_time!(x::CuArray{F,3}, xp::CuArray{F,3},
                               E::CuVector{F}, Ep::CuVector{F},
                               tgt::CuArray{F,3}, pat::CuArray{F,3}, ov::CuArray{F,3},
                               β::F, ra::CuVector{F}, Nf::F, ss::F,
                               max_steps::Int = 10000)::Int
    # Count steps before average φ drops below threshold
    phi_running = vec(mean(tgt .* x, dims=1)) ./ Nf  # Initial φ
    
    trapped_count = 0
    for step in 1:max_steps
        mc_step_single_T!(x, xp, E, Ep, pat, ov, β, ra, Nf, ss)
        phi_running = vec(mean(tgt .* x, dims=1)) ./ Nf
        
        if mean(phi_running) < ESCAPE_THRESHOLD
            trapped_count = step
            break
        end
    end
    
    return trapped_count > 0 ? trapped_count : max_steps  # Cap at max_steps if still trapped
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 80)
    println("LSR Boundary Characterization v5 (Local Minimum Destruction Study)")
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
    @printf("Focus α values: %d (coarse study for boundary characterization)\n", n_alpha)
    @printf("N range: %d – %d,  P range: %d – %d\n", extrema(Ns)..., extrema(Ps)...)
    @printf("T grid: %d log-spaced points (0.05 → 2.5), dense near T_c\n", n_T)
    @printf("Trials: %d,  Sampling: %d steps/T,  Heating: %d init + %d per step\n",
            N_TRIALS, N_SAMP, N_EQ_INIT, N_EQ_STEP)
    println()

    # ── Open CSV for streaming ──
    csv_file = "lsr_boundary_v5.csv"
    csv_h = open(csv_file, "w")
    write(csv_h, "alpha,T,E_mean,E_std,phi_mean,phi_std,phi_min,phi_max,tau_esc,accept_rate\n")
    flush(csv_h)

    # ── Allocate GPU arrays (reuse across T within each α) ──
    println("Processing α values...")
    
    for (i, α) in enumerate(alpha_vec)
        N = Ns[i]; P = Ps[i]; Nf = F(N)
        
        @printf("\n[%d/%d] α=%.2f, N=%d, P=%d\n", i, n_alpha, α, N, P)

        # Generate patterns (once per α)
        Random.seed!(42 + i)
        p_cpu = randn(F, N, P, N_TRIALS)
        for t in 1:N_TRIALS, j in 1:P
            c = @view p_cpu[:, j, t]
            c .*= sqrt(Nf) / norm(c)
        end
        pat_g = CuArray(p_cpu)
        tgt_g = CuArray(p_cpu[:, 1:1, :])  # [N, 1, N_TRIALS]

        # Working arrays
        x_g = CUDA.zeros(F, N, 1, N_TRIALS)
        xp_g = CUDA.zeros(F, N, 1, N_TRIALS)
        ov_g = CUDA.zeros(F, P, 1, N_TRIALS)
        E_g = CUDA.zeros(F, N_TRIALS)
        Ep_g = CUDA.zeros(F, N_TRIALS)
        ra_g = CUDA.zeros(F, N_TRIALS)
        ss = adaptive_ss(N)

        # Accumulators for sampling
        E_acc = CUDA.zeros(F, N_TRIALS)
        phi_acc = CUDA.zeros(F, N_TRIALS)
        accept_acc = CUDA.zeros(F, 1)
        
        # Heating protocol: initialize at coldest T, then propagate
        T_cold = T_vec[1]
        β_cold = F(1.0) / T_cold
        
        # Initialize near target at T_cold
        @printf("  Initializing and heavy equilibration at T=%.4f...\n", T_cold)
        x_g .= tgt_g .+ F(0.05) .* CUDA.randn(F, N, 1, N_TRIALS)
        nrm = sqrt.(sum(x_g .^ 2, dims=1))
        x_g .= sqrt(Nf) .* x_g ./ nrm
        compute_energy_lsr_batched!(E_g, x_g, pat_g, ov_g, Nf)

        # Heavy equilibration at coldest T
        prog = Progress(N_EQ_INIT, desc="  Init EQ: ")
        for step in 1:N_EQ_INIT
            mc_step_single_T!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β_cold, ra_g, Nf, ss)
            next!(prog)
        end
        finish!(prog)
        CUDA.synchronize()

        # ── Loop over T (heating protocol) ──
        prog = Progress(n_T, desc="  T sweep : ")
        for j in 1:n_T
            T = T_vec[j]
            β = F(1.0) / T

            # Re-equilibrate at this T
            n_eq = (j == 1) ? 0 : N_EQ_STEP  # First T already equilibrated
            for step in 1:n_eq
                mc_step_single_T!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, ra_g, Nf, ss)
            end

            # Reset accumulators
            E_acc .= zero(F)
            phi_acc .= zero(F)
            accept_acc .= zero(F)

            # Sample observables
            for step in 1:N_SAMP
                a_rate = mc_step_single_T!(x_g, xp_g, E_g, Ep_g, pat_g, ov_g, β, ra_g, Nf, ss)
                E_acc .+= E_g
                phi_acc .+= vec(sum(tgt_g .* x_g, dims=1)) ./ Nf
                accept_acc[1] += a_rate
            end
            CUDA.synchronize()

            # Measure escape time (separate initialized state)
            x_esc = copy(x_g)
            E_esc = copy(E_g)
            tau_esc = measure_escape_time!(x_esc, xp_g, E_esc, Ep_g, tgt_g, pat_g, ov_g, β, ra_g, Nf, ss)

            # Collect statistics
            E_mean = mean(Array(E_acc)) / N_SAMP
            E_std = std(Array(E_acc)) / N_SAMP
            phi_data = Array(phi_acc) ./ N_SAMP
            phi_mean = mean(phi_data)
            phi_std = std(phi_data)
            phi_min = minimum(phi_data)
            phi_max = maximum(phi_data)
            accept_rate = Float32(accept_acc[1]) / N_SAMP

            # Stream result to CSV
            @printf(csv_h, "%.2f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.4f\n",
                    α, T, E_mean, E_std, phi_mean, phi_std, phi_min, phi_max, tau_esc, accept_rate)
            flush(csv_h)

            next!(prog)
        end
        finish!(prog)

        # Free GPU memory for this α
        CUDA.unsafe_free!(pat_g)
        CUDA.unsafe_free!(tgt_g)
        CUDA.unsafe_free!(x_g); CUDA.unsafe_free!(xp_g)
        CUDA.unsafe_free!(ov_g); CUDA.unsafe_free!(E_g); CUDA.unsafe_free!(Ep_g)
        CUDA.unsafe_free!(ra_g); CUDA.unsafe_free!(E_acc); CUDA.unsafe_free!(phi_acc)
        GC.gc()

        @printf("  ✓ α=%.2f complete\n", α)
    end

    close(csv_h)
    println("\n" * "=" ^ 80)
    @printf("CSV saved: %s\n", csv_file)
    @printf("Observables: E_mean, E_std, φ_mean, φ_std, φ_min, φ_max, τ_esc, accept_rate\n")
    @printf("Ready for analysis: minimum collapse, barrier height, critical T_c(α)\n")
    println("=" ^ 80)
end

main()
