#=
GPU-Accelerated LSE Basin Stability Test (v1)
────────────────────────────────────────────────────────────────────────
Protocol: Basin of Attraction / Retrieval Test
- Random initial states: φ_initial ∈ [0.75, 1.0]
- Relaxation dynamics: n_steps = 2^14 = 16384 steps
- High statistics: n_trials = 1024 trials
- Question: Does initialization with φ>0.75 converge to ξ¹ (stable basin)?
- Output: basin_stab_LSE_v2.csv
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

# ──────────────── Precision Settings ────────────────
const USE_FLOAT16 = false             # Set to true for half precision (50% memory)
const F = USE_FLOAT16 ? Float16 : Float32

# ──────────────── Basin Stability Parameters ────────────────
const betanet     = F(1.0)
const PHI_MIN     = F(0.75)           # Minimum initial alignment
const PHI_MAX     = F(1.0)            # Maximum initial alignment
const MAX_N_TRIALS = 1024             # Maximum trials (at α_min)
const MIN_N_TRIALS = 64               # Minimum trials (at α_max)
const N_STEPS     = 2^14              # 2^14 = 16384 relaxation steps

const alpha_vec = collect(F(0.01):F(0.01):F(0.55))  # 55 values
const T_vec     = F.(10 .^ range(-2, log10(2.5), length=50))  # log-spaced
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)
const MIN_PAT   = 50000
const MAX_PAT   = 500000
const ind       = 10                  # Power-law index for pattern count scaling

# Target memory per chunk: use up to 90% of 50GB = 45GB
const TARGET_MEM_PER_CHUNK_GB = 45.0

# Adaptive pattern count: P(α) follows power-law distribution
# ind=1 → linear spacing (MIN_PAT to MAX_PAT)
# ind=2 → quadratic spacing (more patterns at high α)
# ind>1 → concentrated toward MAX_PAT
const n_patterns_vec = range(MIN_PAT^(1/ind), MAX_PAT^(1/ind), length=n_alpha) .^ ind

# Adaptive trial count: decreases linearly with α
# More trials for low α (small P), fewer for high α (large P)
function n_trials_for_alpha(idx::Int)
    t = (idx - 1) / (n_alpha - 1)  # 0 at idx=1, 1 at idx=n_alpha
    trials = round(Int, MAX_N_TRIALS * (1 - t) + MIN_N_TRIALS * t)
    return max(MIN_N_TRIALS, trials)
end

const n_trials_vec = [n_trials_for_alpha(i) for i in 1:n_alpha]

# Adaptive step size
adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── LSE Energy (batched over trials) ────────────────
function compute_energy_lse!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F)
    # Batched gemm: overlap[p,j,t] = Σ_n patterns[n,p,t] * x[n,j,t]
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)

    # Log-sum-exp trick
    @. overlap = -betanet * (Nf - overlap)
    m = maximum(overlap, dims=1)
    @. overlap = exp(overlap - m)
    s = sum(overlap, dims=1)
    E .= vec(@. -(m + log(s)) / betanet)
    return nothing
end

# ──────────────── MC Step ────────────────
function mc_step!(x::CuArray{F,3}, xp::CuArray{F,3},
                  E::CuVector{F}, Ep::CuVector{F},
                  pat::CuArray{F,3}, ov::CuArray{F,3},
                  β::CuVector{F}, ra::CuVector{F},
                  Nf::F, ss::F)
    # Propose on sphere
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    # Proposed energy
    compute_energy_lse!(Ep, xp, pat, ov, Nf)

    # Metropolis accept/reject
    CUDA.rand!(ra)
    acc = @. ra < exp(-(β * (Ep - E)))
    a3 = reshape(acc, 1, n_T, N_TRIALS)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Random initialization with controlled φ ────────────────
function initialize_random_alignment!(x::Array{F,3}, target::Array{F,3}, N::Int,
                                     phi_min::F, phi_max::F)
    # x: [N × n_T × N_TRIALS]
    # target: [N × 1 × N_TRIALS]
    # Each trial gets random φ ∈ [phi_min, phi_max]

    tgt = target[:, 1, :]  # [N × N_TRIALS]

    for t in 1:N_TRIALS
        for j in 1:n_T
            # Random alignment in [phi_min, phi_max]
            phi_init = phi_min + (phi_max - phi_min) * rand(F)

            # Random perpendicular vector
            x_perp = randn(F, N)
            # Gram-Schmidt orthogonalization
            overlap = dot(tgt[:, t], x_perp) / N
            x_perp .-= overlap .* tgt[:, t]
            x_perp ./= norm(x_perp)

            # Construct state: x = phi*target + sqrt(1-phi²)*sqrt(N)*perp
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
    println("LSE Basin Stability Test – GPU v1")
    println("=" ^ 70)
    println("Protocol: Retrieval / Basin of Attraction")
    println("  Initial alignment: φ ∈ [$PHI_MIN, $PHI_MAX] (random per trial)")
    println("  Precision: $(sizeof(F)*8)-bit float ($(USE_FLOAT16 ? "half" : "single"))")
    println("  Relaxation: $N_STEPS steps")
    println("  Trials: $MAX_N_TRIALS (α_min) → $MIN_N_TRIALS (α_max)")
    println("=" ^ 70)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # Compute N(α), P(α) for all α
    Ns = Int[]; Ps = Int[]
    for (idx, α) in enumerate(alpha_vec)
        Pt = n_patterns_vec[idx]
        N  = max(round(Int, log(Pt) / α), 2)
        push!(Ns, N); push!(Ps, round(Int, Pt))
    end
    @printf("N range: %d – %d,  P range: %d – %d\n", extrema(Ns)..., extrema(Ps)...)
    @printf("Trials range: %d – %d\n", extrema(n_trials_vec)...)
    @printf("Grid: %d α × %d T,  steps: %d\n\n",
            n_alpha, n_T, N_STEPS)

    # ── Calculate memory per α and chunk size ──
    println("Calculating memory requirements...")

    # Calculate memory for each α
    mem_per_alpha_vec = Float64[]
    for (idx, α) in enumerate(alpha_vec)
        N = Ns[idx]
        P = Ps[idx]
        n_trials = n_trials_vec[idx]
        n_chains = n_T * n_trials

        mem = (N*P*n_trials + N*n_trials + 2*N*n_T*n_trials +
               P*n_T*n_trials + 5*n_chains) * sizeof(F)  # includes β_gpu, ra
        push!(mem_per_alpha_vec, mem)
    end

    max_mem_per_alpha = maximum(mem_per_alpha_vec)
    max_mem_idx = argmax(mem_per_alpha_vec)

    @printf("Memory per α: %.2f GB (min) to %.2f GB (max at α=%.2f)\n",
            minimum(mem_per_alpha_vec)/1e9, max_mem_per_alpha/1e9, alpha_vec[max_mem_idx])
    @printf("Target memory per chunk: %.1f GB\n", TARGET_MEM_PER_CHUNK_GB)

    # Determine chunk size
    available_mem = CUDA.available_memory() * 0.85
    target_mem = min(available_mem, TARGET_MEM_PER_CHUNK_GB * 1e9)
    chunk_size = max(1, floor(Int, target_mem / max_mem_per_alpha))

    @printf("Available GPU memory: %.2f GB\n", available_mem/1e9)
    @printf("Chunk size: %d α value(s) at a time\n\n", chunk_size)

    # ── Write CSV header ──
    csv_file = "basin_stab_LSE_v2.csv"
    open(csv_file, "w") do f
        write(f, "alpha")
        for T in T_vec
            write(f, @sprintf(",T%.4f", T))
        end
        write(f, "\n")
    end

    # ── Process α values in chunks ──
    phi_grid = zeros(Float64, n_alpha, n_T)
    t_total = 0.0

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
            P = Ps[global_i]
            n_trials = n_trials_vec[global_i]
            n_chains = n_T * n_trials
            Nf = F(N)

            # Generate normalized patterns (reproducible seed)
            Random.seed!(42 + global_i)
            p_cpu = randn(F, N, P, n_trials)
            for t in 1:n_trials, j in 1:P
                c = @view p_cpu[:, j, t]
                c .*= sqrt(Nf) / norm(c)
            end

            pats_g[local_i]  = CuArray(p_cpu)
            tgts_g[local_i]  = CuArray(p_cpu[:, 1:1, :])
            xs_g[local_i]    = CUDA.zeros(F, N, n_T, n_trials)
            xps_g[local_i]   = CUDA.zeros(F, N, n_T, n_trials)
            ovs_g[local_i]   = CUDA.zeros(F, P, n_T, n_trials)
            Es_g[local_i]    = CUDA.zeros(F, n_chains)
            Eps_g[local_i]   = CUDA.zeros(F, n_chains)
            phis_g[local_i]  = CUDA.zeros(F, n_chains)
            β_gpus[local_i]  = CuVector{F}(repeat(F.(1 ./ T_vec), n_trials))
            ras[local_i]     = CUDA.zeros(F, n_chains)
            ssvec[local_i]   = adaptive_ss(N)
        end
        CUDA.synchronize()
        println("Done.")

        # ── Initialize with random φ ∈ [PHI_MIN, PHI_MAX] ──
        println("Initializing with random alignment φ ∈ [$PHI_MIN, $PHI_MAX]...")
        for (local_i, global_i) in enumerate(chunk_indices)
            N = Ns[global_i]
            n_trials = n_trials_vec[global_i]
            Nf = F(N)

            # Initialize on CPU with controlled alignment
            x_cpu = zeros(F, N, n_T, n_trials)
            tgt_cpu = Array(tgts_g[local_i])
            initialize_random_alignment!(x_cpu, tgt_cpu, N, PHI_MIN, PHI_MAX)

            # Transfer to GPU
            xs_g[local_i] .= CuArray(x_cpu)
            compute_energy_lse!(Es_g[local_i], xs_g[local_i], pats_g[local_i],
                               ovs_g[local_i], Nf)
        end
        CUDA.synchronize()
        println("Done.")

        # ── Relaxation dynamics ──
        println("Relaxation dynamics ($N_STEPS steps)...")
        for local_i in 1:n_chunk
            phis_g[local_i] .= zero(F)
        end

        t0 = time()
        prog = Progress(N_STEPS, desc="Relaxation: ")
        for step in 1:N_STEPS
            for (local_i, global_i) in enumerate(chunk_indices)
                Nf = F(Ns[global_i])
                mc_step!(xs_g[local_i], xps_g[local_i], Es_g[local_i], Eps_g[local_i],
                         pats_g[local_i], ovs_g[local_i], β_gpus[local_i], ras[local_i],
                         Nf, ssvec[local_i])
                # Accumulate alignment
                phis_g[local_i] .+= vec(sum(tgts_g[local_i] .* xs_g[local_i], dims=1)) ./ Nf
            end
            next!(prog)
        end
        finish!(prog)
        CUDA.synchronize()
        t_chunk = time() - t0
        t_total += t_chunk
        @printf("Chunk relaxation: %.1f s (%.2f ms/step)\n", t_chunk, 1000*t_chunk/N_STEPS)

        # ── Collect results for this chunk ──
        println("Collecting chunk results...")
        for (local_i, global_i) in enumerate(chunk_indices)
            n_trials = n_trials_vec[global_i]
            phi_avg = Array(phis_g[local_i]) ./ N_STEPS
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
        pats_g = nothing
        tgts_g = nothing
        xs_g = nothing
        xps_g = nothing
        ovs_g = nothing
        Es_g = nothing
        Eps_g = nothing
        phis_g = nothing
        β_gpus = nothing
        ras = nothing
        GC.gc()
        CUDA.reclaim()
    end

    println("\n" * "=" ^ 70)
    println("CSV saved: $csv_file")
    println()

    # ── Sample output ──
    println("Sample data (basin stability):")
    j_mid = n_T ÷ 2
    for idx in [1, n_alpha÷2, n_alpha]
        @printf("  α=%.2f (N=%d, P=%d, trials=%d): φ(T=%.3f)=%.4f, φ(T=%.3f)=%.4f, φ(T=%.3f)=%.4f\n",
                alpha_vec[idx], Ns[idx], Ps[idx], n_trials_vec[idx],
                T_vec[1], phi_grid[idx, 1],
                T_vec[j_mid], phi_grid[idx, j_mid],
                T_vec[end], phi_grid[idx, end])
    end
    println()
    total_mc_steps = sum(n_trials_vec) * n_T * N_STEPS
    @printf("Total GPU time: %.1f s\n", t_total)
    @printf("Total MC steps: %.2e\n", Float64(total_mc_steps))
    @printf("Average: %.2f ms per MC step\n", 1000*t_total/total_mc_steps)
    println("=" ^ 70)
    println("Interpretation:")
    println("  φ ≈ 1 (blue): Basin stable → successful retrieval")
    println("  φ < 0.75 (red): Basin unstable → retrieval fails")
    println("=" ^ 70)
end

main()
