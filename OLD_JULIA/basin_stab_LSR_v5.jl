#=
GPU-Accelerated LSR Basin Stability Test (v5)
────────────────────────────────────────────────────────────────────────
v5: Poisson pattern reduction + temperature-dependent step size
    σ(N,T) = 2.4·T/√N from v6.

Key idea: The [·]_+ hard wall at φ_c ≈ 0.707 means only K active noise
patterns contribute to the energy near the target.
    K ~ Poisson(λ),   λ = M · P(z > φ_c√N)
                         ≈ exp(N(α − φ_c²/2)) / (φ_c · √(2πN))
using the Mill's ratio approximation for the Gaussian tail.
Instead of storing all M = exp(Nα) patterns, we:
  1. Sample K ~ Poisson(λ) active pattern count per trial
  2. Sample each active pattern's overlap from truncated N(0,1) | z > φ_c√N
     using Robert (1995) exponential-proposal method
  3. Construct full N-dim vectors via Gram–Schmidt against the target
This replaces the [N × M × N_tr] array with [N × (1+K_max) × N_tr].

Background energy: The pruned (M−K) far-away patterns still provide an
energy floor S_bg at any point on the sphere. Without this, the MC cannot
escape the target's basin even when it should (energy → +∞ below φ_c).
    S_bg = λ · b / (φ_c · N)
is added to the energy sum, representing the expected contribution from
far-away random patterns at an arbitrary direction.

Protocol: Basin of Attraction / Retrieval Test
- Random initial states: φ_initial ∈ [0.75, 1.0]
- Equilibration: N_EQ = 2^14 = 16384 steps (unmeasured)
- Sampling: N_SAMP = 2^12 = 4096 steps (φ measured here only)
- Trials: 512 (α ≤ 0.50), tapering to 449 (α = 0.55)
- LSR energy with b = 2 + √2 (hard wall at φ_c ≈ 0.707)
- Output: basin_stab_LSR_v5.csv
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
const phi_c_lsr   = Float64((b_lsr - 1) / b_lsr)  # ≈ 0.707
const PHI_MIN     = F(0.75)
const PHI_MAX     = F(1.0)
const MAX_N_TRIALS = 512
const N_EQ        = 2^14              # 16384 equilibration steps (unmeasured)
const N_SAMP      = 2^12              # 4096 sampling steps (measured)

const alpha_vec = collect(F(0.01):F(0.01):F(0.55))  # 55 values
const T_vec     = collect(F(0.025):F(0.05):F(2.0))   # 40 values
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)
const MIN_PAT   = 20000
const MAX_PAT   = 500000
const ind       = 10

const TARGET_MEM_PER_CHUNK_GB = 42.5
const INF_ENERGY = F(1e30)

# M(α) from v3 schedule (power-law interpolation)
const n_patterns_vec = range(MIN_PAT^(1/ind), MAX_PAT^(1/ind), length=n_alpha) .^ ind

# Trial schedule: 512 for α ≤ 0.50, linear taper to 449 at α = 0.55
function n_trials_for_alpha(idx::Int)
    α = Float64(alpha_vec[idx])
    if α <= 0.50
        return 512
    else
        t = (α - 0.50) / 0.05
        return round(Int, 512 * (1 - t) + 449 * t)
    end
end

const n_trials_vec = [n_trials_for_alpha(i) for i in 1:n_alpha]

# ──────────────── Poisson pattern reduction helpers ────────────────

# Sample from Poisson(λ)
function rand_poisson(λ::Float64)
    if λ == 0.0
        return 0
    elseif λ < 30.0
        # Knuth's direct method
        L = exp(-λ)
        k = 0
        p = 1.0
        while p > L
            k += 1
            p *= rand()
        end
        return k - 1
    else
        # Normal approximation for large λ
        return max(0, round(Int, λ + sqrt(λ) * randn()))
    end
end

# Sample from N(0,1) truncated to (a, ∞) — Robert (1995) exponential proposal
function sample_truncated_normal(a::Float64)
    α_opt = (a + sqrt(a^2 + 4.0)) / 2.0
    while true
        z = a - log(rand()) / α_opt
        ρ = exp(-(z - α_opt)^2 / 2.0)
        if rand() < ρ
            return z
        end
    end
end

# v6: Temperature-dependent step size σ(N,T) = 2.4·T/√N
# Returns a [1 × n_T × 1] CuArray for broadcasting against [N × n_T × n_trials]
function make_ss_gpu(N::Int)
    ss_cpu = F.(2.4 .* T_vec ./ sqrt(F(N)))
    return CuArray(reshape(ss_cpu, 1, n_T, 1))
end

# ──────────────── LSR Energy (batched over trials) ────────────────
function compute_energy_lsr!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F, S_bg::F)
    Nb = Nf / b_lsr
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)

    @. overlap = max(zero(F), one(F) - b_lsr + b_lsr * overlap / Nf)
    s = sum(overlap, dims=1) .+ S_bg
    E .= vec(@. ifelse(s > zero(F), -Nb * log(s), INF_ENERGY))
    return nothing
end

# ──────────────── MC Step ────────────────
function mc_step!(x::CuArray{F,3}, xp::CuArray{F,3},
                  E::CuVector{F}, Ep::CuVector{F},
                  pat::CuArray{F,3}, ov::CuArray{F,3},
                  β::CuVector{F}, ra::CuVector{F},
                  Nf::F, ss::CuArray{F,3}, S_bg::F)
    # Propose on sphere
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    # Proposed energy
    compute_energy_lsr!(Ep, xp, pat, ov, Nf, S_bg)

    # Metropolis accept/reject (reject if basin escape)
    CUDA.rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-(β * (Ep - E))))
    n_trials = length(β) ÷ n_T
    a3 = reshape(acc, 1, n_T, n_trials)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Random initialization with controlled φ ────────────────
function initialize_random_alignment!(x::Array{F,3}, target::Array{F,3}, N::Int,
                                     phi_min::F, phi_max::F)
    tgt = target[:, 1, :]
    n_trials = size(x, 3)

    for t in 1:n_trials
        for j in 1:n_T
            phi_init = phi_min + (phi_max - phi_min) * rand(F)

            x_perp = randn(F, N)
            overlap = dot(tgt[:, t], x_perp) / N
            x_perp .-= overlap .* tgt[:, t]
            x_perp ./= norm(x_perp)

            x[:, j, t] .= phi_init .* tgt[:, t] .+
                          sqrt(1 - phi_init^2) .* sqrt(F(N)) .* x_perp
        end
    end
    return nothing
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 70)
    println("LSR Basin Stability Test – GPU v5 (b=$(round(b_lsr, digits=3)))")
    println("  Poisson reduction + background floor S_bg + σ(N,T) = 2.4·T/√N")
    println("  K ~ Poisson(λ), S_bg = λ·b/(φ_c·N),  λ = exp(N(α−φ_c²/2))/(φ_c√(2πN))")
    println("=" ^ 70)
    println("Protocol: Retrieval / Basin of Attraction")
    println("  Initial alignment: φ ∈ [$PHI_MIN, $PHI_MAX] (random per trial)")
    println("  LSR hard wall: φ_c = $(round(phi_c_lsr, digits=4))")
    println("  Precision: $(sizeof(F)*8)-bit float ($(USE_FLOAT16 ? "half" : "single"))")
    println("  Equilibration: $N_EQ steps (unmeasured)")
    println("  Sampling: $N_SAMP steps (φ measured)")
    println("  Trials: $MAX_N_TRIALS (α ≤ 0.50), tapering at high α")
    println("=" ^ 70)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # ── Compute N(α), M(α) from v3 schedule ──
    Ns = Int[]; Ms = Float64[]
    for (idx, α) in enumerate(alpha_vec)
        M_v3 = n_patterns_vec[idx]
        N = max(round(Int, log(M_v3) / α), 2)
        push!(Ns, N)
        push!(Ms, M_v3)
    end
    @printf("N range: %d – %d,  M range: %d – %d\n",
            extrema(Ns)..., round(Int, minimum(Ms)), round(Int, maximum(Ms)))
    @printf("Trials range: %d – %d\n", extrema(n_trials_vec)...)
    @printf("Grid: %d α × %d T,  MC: %d eq + %d samp\n\n",
            n_alpha, n_T, N_EQ, N_SAMP)

    # ── Compute λ and S_bg (background energy floor) per α ──
    # λ = M · P(z > φ_c√N) ≈ exp(N(α − φ_c²/2)) / (φ_c · √(2πN))  (Mill's ratio)
    half_phic2 = phi_c_lsr^2 / 2.0
    S_bg_vec = Vector{Float64}(undef, n_alpha)
    λ_vec = Vector{Float64}(undef, n_alpha)
    for (idx, α) in enumerate(alpha_vec)
        N = Ns[idx]
        α_f = Float64(α)
        a = phi_c_lsr * sqrt(Float64(N))       # truncation point φ_c√N
        λ = exp(N * (α_f - half_phic2)) / (a * sqrt(2π))  # Mill's ratio
        λ_vec[idx] = λ
        S_bg_vec[idx] = λ * Float64(b_lsr) / (phi_c_lsr * N)
    end

    # ── Poisson statistics preview ──
    println("Poisson pattern reduction (φ_c²/2 = $(round(half_phic2, digits=4))):")
    for idx in [1, n_alpha÷4, n_alpha÷2, 3*n_alpha÷4, n_alpha]
        N = Ns[idx]
        α_f = Float64(alpha_vec[idx])
        @printf("  α=%.2f: N=%d, λ=%.1f, S_bg=%.3f\n",
                alpha_vec[idx], N, λ_vec[idx], S_bg_vec[idx])
    end
    println()

    # ── Generate patterns with Poisson reduction ──
    println("Generating patterns with Poisson reduction...")
    pats_cpu = Vector{Array{F,3}}(undef, n_alpha)
    K_maxs = Int[]

    for (idx, α) in enumerate(alpha_vec)
        N = Ns[idx]
        n_trials = n_trials_vec[idx]
        Nf = F(N)
        sqrtNf = sqrt(Nf)

        # Poisson mean: λ = exp(N(α − φ_c²/2)) / (φ_c√N · √(2π))  (Mill's ratio)
        α_f = Float64(α)
        a_trunc = phi_c_lsr * sqrt(Float64(N))  # truncation point φ_c√N
        λ = exp(N * (α_f - half_phic2)) / (a_trunc * sqrt(2π))

        Random.seed!(42 + idx)

        # Sample K for each trial
        Ks = [rand_poisson(λ) for _ in 1:n_trials]
        K_max = maximum(Ks; init=0)
        push!(K_maxs, K_max)
        P_total = 1 + K_max  # target + active noise patterns

        p_cpu = zeros(F, N, P_total, n_trials)

        for t in 1:n_trials
            # Generate target pattern (ξ¹) on S^{N-1}(√N)
            target = randn(F, N)
            target .*= sqrtNf / norm(target)
            p_cpu[:, 1, t] = target

            # Generate active noise patterns with prescribed overlap
            for k in 1:Ks[t]
                z = sample_truncated_normal(a_trunc)
                phi = F(clamp(z / sqrt(Float64(N)), phi_c_lsr, 1.0 - 1e-6))

                # Gram–Schmidt: random vector orthogonal to target
                u = randn(F, N)
                u .-= (dot(u, target) / dot(target, target)) .* target
                u ./= norm(u)

                # ξ^μ = φ·target + √(N(1-φ²))·û_perp
                p_cpu[:, 1 + k, t] = phi .* target .+ sqrt(Nf * (1 - phi^2)) .* u
            end
            # Remaining columns stay zero (zero-padded → zero energy contribution)
        end

        pats_cpu[idx] = p_cpu
    end

    @printf("K_max range: %d – %d (pattern array sizes: 1+K_max)\n",
            minimum(K_maxs), maximum(K_maxs))
    println("Pattern generation complete.\n")

    # ── Calculate memory per α and chunk size ──
    println("Calculating memory requirements...")

    mem_per_alpha_vec = Float64[]
    for (idx, α) in enumerate(alpha_vec)
        N = Ns[idx]
        P = size(pats_cpu[idx], 2)  # 1 + K_max
        n_trials = n_trials_vec[idx]
        n_chains = n_T * n_trials

        mem = (N*P*n_trials + N*n_trials + 2*N*n_T*n_trials +
               P*n_T*n_trials + 5*n_chains + n_T) * sizeof(F)
        push!(mem_per_alpha_vec, mem)
    end

    max_mem_per_alpha = maximum(mem_per_alpha_vec)
    max_mem_idx = argmax(mem_per_alpha_vec)

    @printf("Memory per α: %.2f MB (min) to %.2f MB (max at α=%.2f)\n",
            minimum(mem_per_alpha_vec)/1e6, max_mem_per_alpha/1e6, alpha_vec[max_mem_idx])
    @printf("Target memory per chunk: %.1f GB\n", TARGET_MEM_PER_CHUNK_GB)

    available_mem = CUDA.available_memory() * 0.85
    target_mem = min(available_mem, TARGET_MEM_PER_CHUNK_GB * 1e9)
    chunk_size = max(1, floor(Int, target_mem / max_mem_per_alpha))

    @printf("Available GPU memory: %.2f GB\n", available_mem/1e9)
    @printf("Chunk size: %d α value(s) at a time\n\n", chunk_size)

    # ── Write CSV header ──
    csv_file = "basin_stab_LSR_v5.csv"
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

    for chunk_start in 1:chunk_size:n_alpha
        chunk_end = min(chunk_start + chunk_size - 1, n_alpha)
        chunk_indices = chunk_start:chunk_end
        n_chunk = length(chunk_indices)

        println("\n" * "=" ^ 70)
        @printf("Processing chunk: α indices %d–%d (%d/%d)\n",
                chunk_start, chunk_end, chunk_end, n_alpha)
        println("=" ^ 70)

        # ── Allocate GPU memory for this chunk ──
        println("Allocating GPU memory for chunk...")
        pats_g   = Vector{CuArray{F,3}}(undef, n_chunk)
        tgts_g   = Vector{CuArray{F,3}}(undef, n_chunk)
        xs_g     = Vector{CuArray{F,3}}(undef, n_chunk)
        xps_g    = Vector{CuArray{F,3}}(undef, n_chunk)
        ovs_g    = Vector{CuArray{F,3}}(undef, n_chunk)
        Es_g     = Vector{CuVector{F}}(undef, n_chunk)
        Eps_g    = Vector{CuVector{F}}(undef, n_chunk)
        phis_g   = Vector{CuVector{F}}(undef, n_chunk)
        β_gpus   = Vector{CuVector{F}}(undef, n_chunk)
        ras      = Vector{CuVector{F}}(undef, n_chunk)
        ss_gpus  = Vector{CuArray{F,3}}(undef, n_chunk)
        S_bg_chunk = Vector{F}(undef, n_chunk)

        for (local_i, global_i) in enumerate(chunk_indices)
            N = Ns[global_i]
            P = size(pats_cpu[global_i], 2)
            n_trials = n_trials_vec[global_i]
            n_chains = n_T * n_trials
            Nf = F(N)

            pats_g[local_i]  = CuArray(pats_cpu[global_i])
            tgts_g[local_i]  = CuArray(pats_cpu[global_i][:, 1:1, :])
            xs_g[local_i]    = CUDA.zeros(F, N, n_T, n_trials)
            xps_g[local_i]   = CUDA.zeros(F, N, n_T, n_trials)
            ovs_g[local_i]   = CUDA.zeros(F, P, n_T, n_trials)
            Es_g[local_i]    = CUDA.zeros(F, n_chains)
            Eps_g[local_i]   = CUDA.zeros(F, n_chains)
            phis_g[local_i]  = CUDA.zeros(F, n_chains)
            β_gpus[local_i]  = CuVector{F}(repeat(F.(1 ./ T_vec), n_trials))
            ras[local_i]     = CUDA.zeros(F, n_chains)
            ss_gpus[local_i] = make_ss_gpu(N)
            S_bg_chunk[local_i] = F(S_bg_vec[global_i])
        end
        CUDA.synchronize()
        println("Done.")

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
                               ovs_g[local_i], Nf, S_bg_chunk[local_i])
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
                         Nf, ss_gpus[local_i], S_bg_chunk[local_i])
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
                         Nf, ss_gpus[local_i], S_bg_chunk[local_i])
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
            for (local_i, global_i) in enumerate(chunk_indices)
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
        xs_g = nothing; xps_g = nothing
        ovs_g = nothing; Es_g = nothing
        Eps_g = nothing; phis_g = nothing
        β_gpus = nothing; ras = nothing
        ss_gpus = nothing; S_bg_chunk = nothing
        GC.gc()
        CUDA.reclaim()
    end

    # ── Free CPU pattern arrays ──
    for idx in 1:n_alpha
        pats_cpu[idx] = zeros(F, 0, 0, 0)
    end

    println("\n" * "=" ^ 70)
    println("CSV saved: $csv_file")
    println()

    println("Sample data (basin stability):")
    j_mid = n_T ÷ 2
    for idx in [1, n_alpha÷2, n_alpha]
        @printf("  α=%.2f (N=%d, K_max=%d, trials=%d): φ(T=%.3f)=%.4f, φ(T=%.3f)=%.4f, φ(T=%.3f)=%.4f\n",
                alpha_vec[idx], Ns[idx], K_maxs[idx], n_trials_vec[idx],
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
    println("  Poisson pattern reduction: K ~ Poisson(λ), λ = M·P(z>φ_c√N) via Mill's ratio")
    println("  Background energy floor: S_bg = λ·b/(φ_c·N)")
    println("  Step size: σ(N,T) = 2.4·T/√N")
    println("  Equilibration: $N_EQ steps (transient discarded)")
    println("  Sampling: $N_SAMP steps (steady-state φ measured)")
    println("Interpretation:")
    println("  φ ≈ 1 (blue): Basin stable → successful retrieval")
    println("  φ < 0.75 (red): Basin unstable → retrieval fails")
    println("=" ^ 70)
end

main()
