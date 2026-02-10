#=
GPU-Accelerated LSR Basin Stability Test (v4)
────────────────────────────────────────────────────────────────────────
v4 improvements over v3:

1. Reduced pattern generation using the LSR hard-wall property:
   [1 - b + b·φ^μ]₊ = 0  for φ^μ < φ_c = (b-1)/b ≈ 0.707.
   Only K << P noise patterns with anomalously high overlap survive.
   K ~ exp(N(α - α_th))  where α_th = ½(1-1/b)² ≈ 0.25.

2. Poisson + truncated-normal sampling (O(K) per trial, not O(P)):
   P can now be arbitrarily large (e.g. 10^11) since we never
   generate P-dimensional arrays.  P = exp(α·N) is computed from N.

3. Minimum dimension N_MIN = 50 enforced for all α.
   (v3 had N = 24 at α = 0.55, giving huge finite-size effects.)

4. More trials: 512 for α ≤ 0.50, linear taper to 449 at α = 0.55
   (fits within ~42.5 GB GPU memory budget).

Protocol otherwise identical to v3.
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

# ──────────────── Precision Settings ────────────────
const USE_FLOAT16 = false
const F = USE_FLOAT16 ? Float16 : Float32

# ──────────────── Basin Stability Parameters ────────────────
const b_lsr       = F(2 + sqrt(2))    # ≈ 3.414
const phi_c_lsr   = F((b_lsr - 1) / b_lsr)  # ≈ 0.707
const alpha_th    = Float64(0.5 * (1 - 1/b_lsr)^2)  # ≈ 0.25
const PHI_MIN     = F(0.75)
const PHI_MAX     = F(1.0)
const BASE_N_TRIALS = 512     # trials for α ≤ 0.50
const END_N_TRIALS  = 449     # trials at α = 0.55 (tapered to fit ~42.5 GB)
const N_EQ        = 2^14              # 16384 equilibration steps (unmeasured)
const N_SAMP      = 2^12              # 4096 sampling steps (measured)

# v4: Minimum dimension — ensures N ≥ N_MIN for all α.
# For α ≤ α_th this is irrelevant (N is already large from the old formula).
# For α > α_th the old formula gave N = 24–46; we enforce N ≥ 50.
const N_MIN       = 50

const alpha_vec = collect(F(0.01):F(0.01):F(0.55))  # 55 values
const T_vec     = collect(F(0.025):F(0.05):F(2.0))   # 40 values
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

# v3-compatible pattern count (used only for N computation at low α)
const MIN_PAT   = 20000
const MAX_PAT   = 500000
const ind       = 10
const n_patterns_vec = range(MIN_PAT^(1/ind), MAX_PAT^(1/ind), length=n_alpha) .^ ind

# Target memory per chunk
const TARGET_MEM_PER_CHUNK_GB = 42.5

function n_trials_for_alpha(idx::Int)
    α = Float64(alpha_vec[idx])
    if α <= 0.50
        return BASE_N_TRIALS
    else
        # Linear taper from 512 (α=0.50) to 449 (α=0.55)
        t = (α - 0.50) / 0.05
        return round(Int, BASE_N_TRIALS * (1 - t) + END_N_TRIALS * t)
    end
end

const n_trials_vec = [n_trials_for_alpha(i) for i in 1:n_alpha]

adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── Truncated Normal Sampler ────────────────
# Robert (1995) exponential-proposal method for z > a, a > 0.
# Acceptance rate > 50% for all a ≥ 0.
function rand_truncnorm_above(a::Float64)
    α_opt = 0.5 * (a + sqrt(a^2 + 4.0))
    while true
        z = a - log(rand()) / α_opt          # exponential proposal shifted to a
        rho = exp(-0.5 * (z - α_opt)^2)
        rand() < rho && return z
    end
end

# ──────────────── LSR Energy (identical to v3) ────────────────
const INF_ENERGY = F(1e30)

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

# ──────────────── MC Step (identical to v3) ────────────────
function mc_step!(x::CuArray{F,3}, xp::CuArray{F,3},
                  E::CuVector{F}, Ep::CuVector{F},
                  pat::CuArray{F,3}, ov::CuArray{F,3},
                  β::CuVector{F}, ra::CuVector{F},
                  Nf::F, ss::F)
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lsr!(Ep, xp, pat, ov, Nf)

    CUDA.rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-(β * (Ep - E))))
    n_trials = length(β) ÷ n_T
    a3 = reshape(acc, 1, n_T, n_trials)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Random initialization (identical to v3) ────────────────
function initialize_random_alignment!(x::Array{F,3}, target::Array{F,3}, N::Int,
                                     phi_min::F, phi_max::F)
    tgt = target[:, 1, :]
    n_trials = size(x, 3)
    for t in 1:n_trials
        for j in 1:n_T
            phi_init = phi_min + (phi_max - phi_min) * rand(F)
            x_perp = randn(F, N)
            ov = dot(tgt[:, t], x_perp) / N
            x_perp .-= ov .* tgt[:, t]
            x_perp ./= norm(x_perp)
            x[:, j, t] .= phi_init .* tgt[:, t] .+
                          sqrt(1 - phi_init^2) .* sqrt(F(N)) .* x_perp
        end
    end
    return nothing
end

# ──────────────── v4: Generate reduced pattern set ────────────────
"""
Generate target + K active noise patterns via Poisson + truncated normal.

The strict threshold c_min = φ_c is used: a zero-padded pattern at
overlap exactly φ_c contributes [b(φ_c - φ_c)]₊ = 0, so borderline
patterns are negligible.

Cost: O(K·N) per trial, independent of P.
"""
function generate_reduced_patterns(N::Int, P_float::Float64, n_trials::Int)
    Nf = F(N)
    sqrtN = sqrt(Nf)

    # Strict threshold: exactly φ_c (no safety margin needed for LSR)
    c_min = Float64(phi_c_lsr)
    z_threshold = c_min * sqrt(Float64(N))

    # Tail probability: P(Z > z_threshold)
    p_tail = 0.5 * erfc(z_threshold / sqrt(2.0))
    lambda = (P_float - 1) * p_tail   # Poisson mean

    # Generate targets on S^{N-1}(√N)
    targets = randn(F, N, n_trials)
    for t in 1:n_trials
        targets[:, t] .*= sqrtN / norm(targets[:, t])
    end

    # Sample K per trial from Poisson(λ)
    K_per_trial = zeros(Int, n_trials)
    active_c = Vector{Vector{Float64}}(undef, n_trials)

    for t in 1:n_trials
        # Poisson sampling
        if lambda < 30
            # Direct method for small λ
            K = 0; L = exp(-lambda); p = 1.0
            while true
                p *= rand()
                p < L && break
                K += 1
            end
        else
            # Normal approximation for large λ
            K = max(0, round(Int, lambda + sqrt(lambda) * randn()))
        end
        K_per_trial[t] = K

        # Sample K overlaps from truncated normal z > z_threshold
        c_vals = Vector{Float64}(undef, K)
        for k in 1:K
            z = rand_truncnorm_above(z_threshold)
            c_vals[k] = min(z / sqrt(Float64(N)), 0.9999)
        end
        active_c[t] = c_vals
    end

    K_max = max(0, maximum(K_per_trial))
    Kp1 = K_max + 1

    # Build pattern array [N, Kp1, n_trials]
    patterns = zeros(F, N, Kp1, n_trials)

    for t in 1:n_trials
        patterns[:, 1, t] .= targets[:, t]

        tgt_unit = targets[:, t] ./ sqrtN

        for (k, c) in enumerate(active_c[t])
            v = randn(F, N)
            v .-= dot(v, tgt_unit) * tgt_unit
            v ./= norm(v)
            patterns[:, k + 1, t] .= F(c) .* targets[:, t] .+ sqrt(Nf * F(1 - c^2)) .* v
        end
    end

    return patterns, targets, K_max, K_per_trial, c_min, lambda
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 70)
    println("LSR Basin Stability Test – GPU v4 (reduced patterns)")
    println("=" ^ 70)
    println("Protocol: Retrieval / Basin of Attraction")
    println("  Initial alignment: φ ∈ [$PHI_MIN, $PHI_MAX] (random per trial)")
    println("  LSR hard wall: φ_c = $(round(phi_c_lsr, digits=4))")
    println("  LSR b = $(round(b_lsr, digits=3))")
    println("  α_th = $(round(alpha_th, digits=4))")
    println("  N_MIN = $N_MIN (enforced minimum dimension)")
    println("  Threshold: strict (c_min = φ_c, no safety margin)")
    println("  Precision: $(sizeof(F)*8)-bit float ($(USE_FLOAT16 ? "half" : "single"))")
    println("  Equilibration: $N_EQ steps (unmeasured)")
    println("  Sampling: $N_SAMP steps (φ measured)")
    println("  Trials: $BASE_N_TRIALS (α ≤ 0.50) → $END_N_TRIALS (α = 0.55)")
    println("=" ^ 70)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # ── Compute N(α) with N_MIN enforcement ──
    # For α ≤ α_th: use v3 formula N = log(P)/α  (N is already large)
    # For α > α_th: N = max(N_MIN, v3_formula)
    # P is derived from N: P = exp(α·N)
    Ns = Int[]
    Ps_float = Float64[]   # P can be huge (10^11), store as Float64
    for (idx, α) in enumerate(alpha_vec)
        Pt = n_patterns_vec[idx]
        N_v3 = max(round(Int, log(Pt) / Float64(α)), 2)
        N = max(N_v3, N_MIN)
        push!(Ns, N)
        push!(Ps_float, exp(Float64(α) * N))
    end
    @printf("N range: %d – %d\n", extrema(Ns)...)
    @printf("P range: %.2e – %.2e (derived from N)\n", extrema(Ps_float)...)
    @printf("Trials range: %d – %d\n", extrema(n_trials_vec)...)
    @printf("Grid: %d α × %d T,  MC: %d eq + %d samp\n\n",
            n_alpha, n_T, N_EQ, N_SAMP)

    # ── Pre-compute expected K for each α ──
    println("v4 pattern reduction estimates:")
    println("-" ^ 70)
    @printf("  %-6s %-5s %-10s %-10s %-12s %-10s\n",
            "α", "N", "P", "λ (Poisson)", "K_strict", "N improvement")
    println("-" ^ 70)

    K_expected_vec = Float64[]
    for (idx, α) in enumerate(alpha_vec)
        N = Ns[idx]
        P = Ps_float[idx]
        z_th = Float64(phi_c_lsr) * sqrt(Float64(N))
        p_tail = 0.5 * erfc(z_th / sqrt(2.0))
        lambda = (P - 1) * p_tail
        push!(K_expected_vec, lambda)

        # Also compute v3 N for comparison
        Pt = n_patterns_vec[idx]
        N_v3 = max(round(Int, log(Pt) / Float64(α)), 2)

        if idx % 5 == 1 || idx == n_alpha
            K_str = lambda < 0.01 ? "≈0" : @sprintf("%.0f", lambda)
            N_imp = N > N_v3 ? @sprintf("%d → %d", N_v3, N) : "unchanged"
            @printf("  %-6.2f %-5d %-10.2e %-10.2e %-12s %-10s\n",
                    α, N, P, lambda, K_str, N_imp)
        end
    end
    println("-" ^ 70)
    println()

    # ── Calculate memory per α ──
    println("Calculating memory requirements...")
    mem_per_alpha_vec = Float64[]
    for (idx, _) in enumerate(alpha_vec)
        N = Ns[idx]
        n_trials = n_trials_vec[idx]
        n_chains = n_T * n_trials
        K_est = max(1, ceil(Int, K_expected_vec[idx] +
                    3 * sqrt(max(1.0, K_expected_vec[idx]))))
        Kp1 = K_est + 1

        mem = (N*Kp1*n_trials + N*n_trials + 2*N*n_T*n_trials +
               Kp1*n_T*n_trials + 5*n_chains) * sizeof(F)
        push!(mem_per_alpha_vec, mem)
    end

    max_mem = maximum(mem_per_alpha_vec)
    max_idx = argmax(mem_per_alpha_vec)
    @printf("Memory per α: %.3f GB (min) to %.3f GB (max at α=%.2f)\n",
            minimum(mem_per_alpha_vec)/1e9, max_mem/1e9, alpha_vec[max_idx])

    available_mem = CUDA.available_memory() * 0.85
    target_mem = min(available_mem, TARGET_MEM_PER_CHUNK_GB * 1e9)
    chunk_size = max(1, floor(Int, target_mem / max_mem))
    @printf("Available GPU memory: %.2f GB\n", available_mem/1e9)
    @printf("Chunk size: %d α value(s) at a time\n\n", chunk_size)

    # ── Write CSV header ──
    csv_file = "basin_stab_LSR_v4.csv"
    open(csv_file, "w") do f
        write(f, "alpha")
        for T in T_vec
            write(f, @sprintf(",T%.4f", T))
        end
        write(f, "\n")
    end

    # ── Process α values in chunks ──
    phi_grid = zeros(Float64, n_alpha, n_T)
    t_total_eq = 0.0
    t_total_samp = 0.0
    total_patterns_generated = 0

    for chunk_start in 1:chunk_size:n_alpha
        chunk_end = min(chunk_start + chunk_size - 1, n_alpha)
        chunk_indices = chunk_start:chunk_end
        n_chunk = length(chunk_indices)

        println("\n" * "=" ^ 70)
        @printf("Processing chunk: α indices %d–%d (%d/%d)\n",
                chunk_start, chunk_end, chunk_end, n_alpha)
        println("=" ^ 70)

        # ── Generate reduced patterns & allocate GPU memory ──
        println("Generating reduced pattern sets (Poisson + truncated normal)...")
        pats_g  = Vector{CuArray{F,3}}(undef, n_chunk)
        tgts_g  = Vector{CuArray{F,3}}(undef, n_chunk)
        xs_g    = Vector{CuArray{F,3}}(undef, n_chunk)
        xps_g   = Vector{CuArray{F,3}}(undef, n_chunk)
        ovs_g   = Vector{CuArray{F,3}}(undef, n_chunk)
        Es_g    = Vector{CuVector{F}}(undef, n_chunk)
        Eps_g   = Vector{CuVector{F}}(undef, n_chunk)
        phis_g  = Vector{CuVector{F}}(undef, n_chunk)
        β_gpus  = Vector{CuVector{F}}(undef, n_chunk)
        ras     = Vector{CuVector{F}}(undef, n_chunk)
        ssvec   = Vector{F}(undef, n_chunk)

        for (local_i, global_i) in enumerate(chunk_indices)
            N = Ns[global_i]
            P = Ps_float[global_i]
            n_trials = n_trials_vec[global_i]
            n_chains = n_T * n_trials
            Nf = F(N)

            Random.seed!(42 + global_i)

            p_cpu, tgt_cpu, K_max, K_per_trial, c_min, lambda =
                generate_reduced_patterns(N, P, n_trials)

            Kp1 = K_max + 1
            total_patterns_generated += sum(K_per_trial .+ 1)

            @printf("  α=%.2f: N=%d, P=%.2e, λ=%.1f → K_max=%d, ⟨K⟩=%.1f\n",
                    alpha_vec[global_i], N, P, lambda, K_max, mean(K_per_trial))

            pats_g[local_i]  = CuArray(p_cpu)
            tgts_g[local_i]  = CuArray(reshape(tgt_cpu, N, 1, n_trials))
            xs_g[local_i]    = CUDA.zeros(F, N, n_T, n_trials)
            xps_g[local_i]   = CUDA.zeros(F, N, n_T, n_trials)
            ovs_g[local_i]   = CUDA.zeros(F, Kp1, n_T, n_trials)
            Es_g[local_i]    = CUDA.zeros(F, n_chains)
            Eps_g[local_i]   = CUDA.zeros(F, n_chains)
            phis_g[local_i]  = CUDA.zeros(F, n_chains)
            β_gpus[local_i]  = CuVector{F}(repeat(F.(1 ./ T_vec), n_trials))
            ras[local_i]     = CUDA.zeros(F, n_chains)
            ssvec[local_i]   = adaptive_ss(N)
        end
        CUDA.synchronize()
        println("GPU allocation done.")

        # ── Initialize with random φ ∈ [PHI_MIN, PHI_MAX] ──
        println("Initializing with random alignment φ ∈ [$PHI_MIN, $PHI_MAX]...")
        for (local_i, global_i) in enumerate(chunk_indices)
            N = Ns[global_i]
            n_trials = n_trials_vec[global_i]
            Nf = F(N)

            x_cpu = zeros(F, N, n_T, n_trials)
            tgt_cpu = Array(tgts_g[local_i])
            initialize_random_alignment!(x_cpu, tgt_cpu, N, PHI_MIN, PHI_MAX)

            xs_g[local_i] .= CuArray(x_cpu)
            compute_energy_lsr!(Es_g[local_i], xs_g[local_i], pats_g[local_i],
                               ovs_g[local_i], Nf)
        end
        CUDA.synchronize()
        println("Done.")

        # ── Phase 1: Equilibration (unmeasured) ──
        println("Equilibration ($N_EQ steps, unmeasured)...")
        t0 = time()
        prog = Progress(N_EQ, desc="Equilibration: ")
        for step in 1:N_EQ
            for (local_i, global_i) in enumerate(chunk_indices)
                Nf = F(Ns[global_i])
                mc_step!(xs_g[local_i], xps_g[local_i], Es_g[local_i], Eps_g[local_i],
                         pats_g[local_i], ovs_g[local_i], β_gpus[local_i], ras[local_i],
                         Nf, ssvec[local_i])
            end
            next!(prog)
        end
        finish!(prog)
        CUDA.synchronize()
        t_eq = time() - t0
        t_total_eq += t_eq
        @printf("Equilibration: %.1f s (%.2f ms/step)\n", t_eq, 1000*t_eq/N_EQ)

        # ── Phase 2: Sampling (φ measured) ──
        println("Sampling ($N_SAMP steps, measuring φ)...")
        for local_i in 1:n_chunk
            phis_g[local_i] .= zero(F)
        end

        t0 = time()
        prog = Progress(N_SAMP, desc="Sampling: ")
        for step in 1:N_SAMP
            for (local_i, global_i) in enumerate(chunk_indices)
                Nf = F(Ns[global_i])
                mc_step!(xs_g[local_i], xps_g[local_i], Es_g[local_i], Eps_g[local_i],
                         pats_g[local_i], ovs_g[local_i], β_gpus[local_i], ras[local_i],
                         Nf, ssvec[local_i])
                phis_g[local_i] .+= vec(sum(tgts_g[local_i] .* xs_g[local_i], dims=1)) ./ Nf
            end
            next!(prog)
        end
        finish!(prog)
        CUDA.synchronize()
        t_samp = time() - t0
        t_total_samp += t_samp
        @printf("Sampling: %.1f s (%.2f ms/step)\n", t_samp, 1000*t_samp/N_SAMP)

        # ── Collect results for this chunk ──
        println("Collecting chunk results...")
        for (local_i, global_i) in enumerate(chunk_indices)
            n_trials = n_trials_vec[global_i]
            phi_avg = Array(phis_g[local_i]) ./ N_SAMP
            phi_mat = reshape(phi_avg, n_T, n_trials)
            phi_grid[global_i, :] = vec(mean(phi_mat, dims=2))
        end

        # ── Append chunk results to CSV ──
        open(csv_file, "a") do f
            for (_, global_i) in enumerate(chunk_indices)
                write(f, @sprintf("%.2f", alpha_vec[global_i]))
                for j in 1:n_T
                    write(f, @sprintf(",%.4f", phi_grid[global_i, j]))
                end
                write(f, "\n")
            end
        end
        println("Chunk results appended to CSV.")

        # ── Free GPU memory ──
        pats_g = nothing; tgts_g = nothing
        xs_g = nothing; xps_g = nothing; ovs_g = nothing
        Es_g = nothing; Eps_g = nothing; phis_g = nothing
        β_gpus = nothing; ras = nothing
        GC.gc()
        CUDA.reclaim()
    end

    println("\n" * "=" ^ 70)
    println("CSV saved: $csv_file")
    @printf("\nTotal patterns generated (v4): %.2e\n", Float64(total_patterns_generated))
    println()

    # ── Sample output ──
    println("Sample data (basin stability):")
    j_mid = n_T ÷ 2
    for idx in [1, n_alpha÷2, n_alpha]
        @printf("  α=%.2f (N=%d, P=%.1e): φ(T=%.3f)=%.4f, φ(T=%.3f)=%.4f, φ(T=%.3f)=%.4f\n",
                alpha_vec[idx], Ns[idx], Ps_float[idx],
                T_vec[1], phi_grid[idx, 1],
                T_vec[j_mid], phi_grid[idx, j_mid],
                T_vec[end], phi_grid[idx, end])
    end
    println()
    total_mc_steps = sum(n_trials_vec) * n_T * (N_EQ + N_SAMP)
    @printf("Total GPU time: %.1f s (eq: %.1f + samp: %.1f)\n",
            t_total_eq+t_total_samp, t_total_eq, t_total_samp)
    @printf("Total MC steps: %.2e\n", Float64(total_mc_steps))
    println("=" ^ 70)
    println("Protocol summary:")
    println("  Equilibration: $N_EQ steps (transient discarded)")
    println("  Sampling: $N_SAMP steps (steady-state φ measured)")
    println("  v4: Poisson + truncated-normal pattern generation")
    println("  Strict threshold at φ_c (zero-padded slots killed by [·]₊)")
    println("  N_MIN = $N_MIN enforced for all α")
    println("=" ^ 70)
end

main()
