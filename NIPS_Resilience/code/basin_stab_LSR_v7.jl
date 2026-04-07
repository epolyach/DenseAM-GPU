#=
GPU-Accelerated LSR Basin Stability Test (v7) — PRODUCTION
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSR_v7.jl           # resume from last completed α
  julia basin_stab_LSR_v7.jl --fresh   # overwrite CSVs, start from α=0.01

v7 change vs v6: Simultaneously compute φ and q_EA (Edwards-Anderson
    self-overlap) maps using a two-replica approach.
    φ(α,T) = ⟨target·x/N⟩           (overlap with stored pattern)
    q_EA(α,T) = ⟨x_a·x_b/N⟩         (cross-overlap of two replicas)
    Two replicas share the same disorder (patterns) but evolve with
    independent MC noise.  n_disorder = n_trials ÷ 2 pattern sets.
    φ averaged over both replicas (2 × n_disorder chains).
    q averaged over n_disorder cross-pairs.

v6 feature retained: σ(N,T) = 2.4·T/√N

Protocol: Basin of Attraction / Retrieval Test
- Random initial states: φ_initial ∈ [0.75, 1.0]
- Equilibration: N_EQ = 2^14 = 16384 steps (unmeasured)
- Sampling: N_SAMP = 2^12 = 4096 steps (φ and q measured)
- Adaptive trials: 512 (α_min) → 64 (α_max)
- Two replicas per disorder sample (shared patterns, independent MC)
- LSR energy with b = 2 + √2 (hard wall at φ_c ≈ 0.707)
- Output: basin_stab_LSR_v7.csv  (φ map)
          basin_stab_LSR_v7_q.csv (q map)

Resume: by default, reads existing CSV and continues from the first
    missing α row. Both CSVs are checked; uses min(rows) for safety.
    Pass --fresh to overwrite and restart from scratch.

See also: basin_stab_LSE_v7.jl  (LSE energy variant)
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
const PHI_MIN     = F(0.75)
const PHI_MAX     = F(1.0)
const MAX_N_TRIALS = 512
const MIN_N_TRIALS = 64
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

const n_patterns_vec = range(MIN_PAT^(1/ind), MAX_PAT^(1/ind), length=n_alpha) .^ ind

function n_trials_for_alpha(idx::Int)
    t = (idx - 1) / (n_alpha - 1)
    trials = round(Int, MAX_N_TRIALS * (1 - t) + MIN_N_TRIALS * t)
    return max(MIN_N_TRIALS, trials)
end

const n_trials_vec   = [n_trials_for_alpha(i) for i in 1:n_alpha]
const n_disorder_vec = [n_trials_vec[i] ÷ 2 for i in 1:n_alpha]

# v6: Temperature-dependent step size σ(N,T) = 2.4·T/√N
function make_ss_gpu(N::Int)
    ss_cpu = F.(2.4 .* T_vec ./ sqrt(F(N)))
    return CuArray(reshape(ss_cpu, 1, n_T, 1))
end

# ──────────────── LSR Energy (batched over disorder samples) ────────────────
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

# ──────────────── MC Step ────────────────
function mc_step!(x::CuArray{F,3}, xp::CuArray{F,3},
                  E::CuVector{F}, Ep::CuVector{F},
                  pat::CuArray{F,3}, ov::CuArray{F,3},
                  β::CuVector{F}, ra::CuVector{F},
                  Nf::F, ss::CuArray{F,3})
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lsr!(Ep, xp, pat, ov, Nf)

    CUDA.rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-(β * (Ep - E))))
    n_dis = length(β) ÷ n_T
    a3 = reshape(acc, 1, n_T, n_dis)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Random initialization with controlled φ ────────────────
function initialize_random_alignment!(x::Array{F,3}, target::Array{F,3}, N::Int,
                                     phi_min::F, phi_max::F)
    tgt = target[:, 1, :]
    n_dis = size(x, 3)

    for d in 1:n_dis
        for j in 1:n_T
            phi_init = phi_min + (phi_max - phi_min) * rand(F)

            x_perp = randn(F, N)
            overlap = dot(tgt[:, d], x_perp) / N
            x_perp .-= overlap .* tgt[:, d]
            x_perp ./= norm(x_perp)

            x[:, j, d] .= phi_init .* tgt[:, d] .+
                          sqrt(1 - phi_init^2) .* sqrt(F(N)) .* x_perp
        end
    end
    return nothing
end

# ──────────────── CLI & Resume ────────────────
const FRESH_START = "--fresh" in ARGS

function count_completed_rows(csv_file::String)
    !isfile(csv_file) && return 0
    n = 0
    open(csv_file, "r") do f
        for _ in eachline(f)
            n += 1
        end
    end
    return max(0, n - 1)  # subtract header
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    phi_c_lsr = (b_lsr - 1) / b_lsr

    println("=" ^ 70)
    println("LSR Basin Stability Test – GPU v7 (b=$(round(b_lsr, digits=3)), two-replica q_EA)")
    println("  v7: φ + q_EA maps via two-replica cross-overlap")
    println("  v6 retained: σ(N,T) = 2.4·T/√N")
    println("=" ^ 70)
    println("Protocol: Retrieval / Basin of Attraction")
    println("  Initial alignment: φ ∈ [$PHI_MIN, $PHI_MAX] (random per replica)")
    println("  LSR hard wall: φ_c = $(round(phi_c_lsr, digits=4))")
    println("  Precision: $(sizeof(F)*8)-bit float ($(USE_FLOAT16 ? "half" : "single"))")
    println("  Equilibration: $N_EQ steps (unmeasured)")
    println("  Sampling: $N_SAMP steps (φ and q measured)")
    println("  Trials: $MAX_N_TRIALS (α_min) → $MIN_N_TRIALS (α_max)")
    println("  Two replicas per disorder sample → n_disorder = n_trials ÷ 2")
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
    @printf("Trials range: %d – %d  (disorder: %d – %d)\n",
            extrema(n_trials_vec)..., extrema(n_disorder_vec)...)
    @printf("Grid: %d α × %d T,  MC: %d eq + %d samp\n\n",
            n_alpha, n_T, N_EQ, N_SAMP)

    # ── Calculate memory per α and chunk size ──
    println("Calculating memory requirements...")

    mem_per_alpha_vec = Float64[]
    for (idx, α) in enumerate(alpha_vec)
        N = Ns[idx]
        P = Ps[idx]
        n_dis = n_disorder_vec[idx]
        n_chains = n_T * n_dis

        # patterns[N,P,n_dis] + targets[N,1,n_dis]
        # x_a,x_b[N,n_T,n_dis] + xp(shared)[N,n_T,n_dis] + ov(shared)[P,n_T,n_dis]
        # E_a,E_b,Ep(shared),phi_a,phi_b,q_ab,β,ra: 8×n_chains
        # ss: n_T
        mem = (N*P*n_dis + N*n_dis + 3*N*n_T*n_dis +
               P*n_T*n_dis + 8*n_chains + n_T) * sizeof(F)
        push!(mem_per_alpha_vec, mem)
    end

    max_mem_per_alpha = maximum(mem_per_alpha_vec)
    max_mem_idx = argmax(mem_per_alpha_vec)

    @printf("Memory per α: %.2f GB (min) to %.2f GB (max at α=%.2f)\n",
            minimum(mem_per_alpha_vec)/1e9, max_mem_per_alpha/1e9, alpha_vec[max_mem_idx])
    @printf("Target memory per chunk: %.1f GB\n", TARGET_MEM_PER_CHUNK_GB)

    available_mem = CUDA.available_memory() * 0.85
    target_mem = min(available_mem, TARGET_MEM_PER_CHUNK_GB * 1e9)
    chunk_size = max(1, floor(Int, target_mem / max_mem_per_alpha))

    @printf("Available GPU memory: %.2f GB\n", available_mem/1e9)
    @printf("Chunk size: %d α value(s) at a time\n\n", chunk_size)

    # ── Determine resume point ──
    csv_phi = "basin_stab_LSR_v7.csv"
    csv_q   = "basin_stab_LSR_v7_q.csv"
    n_completed = FRESH_START ? 0 : min(count_completed_rows(csv_phi),
                                        count_completed_rows(csv_q))
    start_idx = n_completed + 1

    if start_idx > n_alpha
        println("All $n_alpha α values already completed in CSV.")
        println("Use --fresh to overwrite and restart.")
        return
    end

    if FRESH_START || n_completed == 0
        for csv_file in (csv_phi, csv_q)
            open(csv_file, "w") do f
                write(f, "alpha")
                for T in T_vec
                    write(f, @sprintf(",T%.4f", T))
                end
                write(f, "\n")
            end
        end
        start_idx = 1
        println(FRESH_START ? "Fresh start (--fresh)." : "No existing CSV found, starting fresh.")
    else
        @printf("Resuming: %d/%d α values completed, continuing from α=%.2f (index %d)\n",
                n_completed, n_alpha, alpha_vec[start_idx], start_idx)
    end

    # ── Process α values in chunks ──
    phi_grid = zeros(Float64, n_alpha, n_T)
    q_grid   = zeros(Float64, n_alpha, n_T)
    t_total_eq = 0.0
    t_total_samp = 0.0

    for chunk_start in start_idx:chunk_size:n_alpha
        chunk_end = min(chunk_start + chunk_size - 1, n_alpha)
        chunk_indices = chunk_start:chunk_end
        n_chunk = length(chunk_indices)

        println("\n" * "=" ^ 70)
        @printf("Processing chunk: α indices %d–%d (%d/%d)\n",
                chunk_start, chunk_end, chunk_end, n_alpha)
        println("=" ^ 70)

        # ── Allocate GPU memory for this chunk ──
        println("Allocating GPU memory for chunk...")
        pats_g  = Vector{CuArray{F,3}}(undef, n_chunk)   # patterns (shared)
        tgts_g  = Vector{CuArray{F,3}}(undef, n_chunk)   # targets (shared)
        xa_g    = Vector{CuArray{F,3}}(undef, n_chunk)   # replica A states
        xb_g    = Vector{CuArray{F,3}}(undef, n_chunk)   # replica B states
        xp_g    = Vector{CuArray{F,3}}(undef, n_chunk)   # proposal (shared workspace)
        ov_g    = Vector{CuArray{F,3}}(undef, n_chunk)   # overlap (shared workspace)
        Ea_g    = Vector{CuVector{F}}(undef, n_chunk)    # energy replica A
        Eb_g    = Vector{CuVector{F}}(undef, n_chunk)    # energy replica B
        Ep_g    = Vector{CuVector{F}}(undef, n_chunk)    # proposed energy (shared)
        phis_g  = Vector{CuVector{F}}(undef, n_chunk)    # φ accumulator (both replicas)
        qs_g    = Vector{CuVector{F}}(undef, n_chunk)    # q accumulator (cross-overlap)
        β_g     = Vector{CuVector{F}}(undef, n_chunk)    # inverse temperatures
        ra_g    = Vector{CuVector{F}}(undef, n_chunk)    # random numbers (shared)
        ss_g    = Vector{CuArray{F,3}}(undef, n_chunk)   # step sizes

        for (local_i, global_i) in enumerate(chunk_indices)
            N = Ns[global_i]
            P = Ps[global_i]
            n_dis = n_disorder_vec[global_i]
            n_chains = n_T * n_dis
            Nf = F(N)

            # Generate patterns (one set per disorder sample)
            Random.seed!(42 + global_i)
            p_cpu = randn(F, N, P, n_dis)
            for d in 1:n_dis, j in 1:P
                c = @view p_cpu[:, j, d]
                c .*= sqrt(Nf) / norm(c)
            end

            pats_g[local_i] = CuArray(p_cpu)
            tgts_g[local_i] = CuArray(p_cpu[:, 1:1, :])
            xa_g[local_i]   = CUDA.zeros(F, N, n_T, n_dis)
            xb_g[local_i]   = CUDA.zeros(F, N, n_T, n_dis)
            xp_g[local_i]   = CUDA.zeros(F, N, n_T, n_dis)
            ov_g[local_i]   = CUDA.zeros(F, P, n_T, n_dis)
            Ea_g[local_i]   = CUDA.zeros(F, n_chains)
            Eb_g[local_i]   = CUDA.zeros(F, n_chains)
            Ep_g[local_i]   = CUDA.zeros(F, n_chains)
            phis_g[local_i] = CUDA.zeros(F, n_chains)
            qs_g[local_i]   = CUDA.zeros(F, n_chains)
            β_g[local_i]    = CuVector{F}(repeat(F.(1 ./ T_vec), n_dis))
            ra_g[local_i]   = CUDA.zeros(F, n_chains)
            ss_g[local_i]   = make_ss_gpu(N)
        end
        CUDA.synchronize()
        println("Done.")

        # ── Initialize both replicas independently ──
        println("Initializing replicas with random alignment φ ∈ [$PHI_MIN, $PHI_MAX]...")
        for (local_i, global_i) in enumerate(chunk_indices)
            N = Ns[global_i]
            n_dis = n_disorder_vec[global_i]
            Nf = F(N)
            tgt_cpu = Array(tgts_g[local_i])

            # Replica A
            Random.seed!(1_000_000 + global_i)
            xa_cpu = zeros(F, N, n_T, n_dis)
            initialize_random_alignment!(xa_cpu, tgt_cpu, N, PHI_MIN, PHI_MAX)
            xa_g[local_i] .= CuArray(xa_cpu)
            compute_energy_lsr!(Ea_g[local_i], xa_g[local_i], pats_g[local_i],
                               ov_g[local_i], Nf)

            # Replica B (different seed → different initial conditions)
            Random.seed!(2_000_000 + global_i)
            xb_cpu = zeros(F, N, n_T, n_dis)
            initialize_random_alignment!(xb_cpu, tgt_cpu, N, PHI_MIN, PHI_MAX)
            xb_g[local_i] .= CuArray(xb_cpu)
            compute_energy_lsr!(Eb_g[local_i], xb_g[local_i], pats_g[local_i],
                               ov_g[local_i], Nf)
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
                # Replica A
                mc_step!(xa_g[local_i], xp_g[local_i], Ea_g[local_i], Ep_g[local_i],
                         pats_g[local_i], ov_g[local_i], β_g[local_i], ra_g[local_i],
                         Nf, ss_g[local_i])
                # Replica B (same patterns, independent MC noise)
                mc_step!(xb_g[local_i], xp_g[local_i], Eb_g[local_i], Ep_g[local_i],
                         pats_g[local_i], ov_g[local_i], β_g[local_i], ra_g[local_i],
                         Nf, ss_g[local_i])
            end
            next!(prog)
        end
        finish!(prog)
        CUDA.synchronize()
        t_eq = time() - t0
        t_total_eq += t_eq
        @printf("Equilibration: %.1f s (%.2f ms/step)\n", t_eq, 1000*t_eq/N_EQ)

        # ── Phase 2: Sampling (φ and q measured) ──
        println("Sampling ($N_SAMP steps, measuring φ and q)...")
        for local_i in 1:n_chunk
            phis_g[local_i] .= zero(F)
            qs_g[local_i]   .= zero(F)
        end

        t0 = time()
        prog = Progress(N_SAMP, desc="Sampling: ")
        for step in 1:N_SAMP
            for (local_i, global_i) in enumerate(chunk_indices)
                Nf = F(Ns[global_i])
                # Replica A
                mc_step!(xa_g[local_i], xp_g[local_i], Ea_g[local_i], Ep_g[local_i],
                         pats_g[local_i], ov_g[local_i], β_g[local_i], ra_g[local_i],
                         Nf, ss_g[local_i])
                # Replica B
                mc_step!(xb_g[local_i], xp_g[local_i], Eb_g[local_i], Ep_g[local_i],
                         pats_g[local_i], ov_g[local_i], β_g[local_i], ra_g[local_i],
                         Nf, ss_g[local_i])
                # φ: accumulate from both replicas
                phis_g[local_i] .+= vec(sum(tgts_g[local_i] .* xa_g[local_i], dims=1)) ./ Nf
                phis_g[local_i] .+= vec(sum(tgts_g[local_i] .* xb_g[local_i], dims=1)) ./ Nf
                # q_EA: cross-overlap of the two replicas
                qs_g[local_i] .+= vec(sum(xa_g[local_i] .* xb_g[local_i], dims=1)) ./ Nf
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
            n_dis = n_disorder_vec[global_i]

            # φ: sum of both replicas over N_SAMP steps → divide by 2*N_SAMP
            phi_avg = Array(phis_g[local_i]) ./ (2 * N_SAMP)
            phi_mat = reshape(phi_avg, n_T, n_dis)
            phi_grid[global_i, :] = vec(mean(phi_mat, dims=2))

            # q: cross-overlap over N_SAMP steps → divide by N_SAMP
            q_avg = Array(qs_g[local_i]) ./ N_SAMP
            q_mat = reshape(q_avg, n_T, n_dis)
            q_grid[global_i, :] = vec(mean(q_mat, dims=2))
        end

        # ── Append chunk results to CSVs ──
        for (csv_file, grid) in ((csv_phi, phi_grid), (csv_q, q_grid))
            open(csv_file, "a") do f
                for (local_i, global_i) in enumerate(chunk_indices)
                    write(f, @sprintf("%.2f", alpha_vec[global_i]))
                    for j in 1:n_T
                        write(f, @sprintf(",%.4f", grid[global_i, j]))
                    end
                    write(f, "\n")
                end
            end
        end
        println("Chunk results appended to CSVs.")

        # ── Free GPU memory ──
        pats_g = nothing; tgts_g = nothing
        xa_g = nothing; xb_g = nothing; xp_g = nothing
        ov_g = nothing; Ea_g = nothing; Eb_g = nothing
        Ep_g = nothing; phis_g = nothing; qs_g = nothing
        β_g = nothing; ra_g = nothing; ss_g = nothing
        GC.gc()
        CUDA.reclaim()
    end

    println("\n" * "=" ^ 70)
    println("CSV saved: $csv_phi  (φ map)")
    println("CSV saved: $csv_q   (q map)")
    println()

    println("Sample data (φ and q_EA):")
    j_mid = n_T ÷ 2
    for idx in [1, n_alpha÷2, n_alpha]
        @printf("  α=%.2f (N=%d, P=%d, disorder=%d):\n",
                alpha_vec[idx], Ns[idx], Ps[idx], n_disorder_vec[idx])
        @printf("    φ: T=%.3f→%.4f,  T=%.3f→%.4f,  T=%.3f→%.4f\n",
                T_vec[1], phi_grid[idx, 1],
                T_vec[j_mid], phi_grid[idx, j_mid],
                T_vec[end], phi_grid[idx, end])
        @printf("    q: T=%.3f→%.4f,  T=%.3f→%.4f,  T=%.3f→%.4f\n",
                T_vec[1], q_grid[idx, 1],
                T_vec[j_mid], q_grid[idx, j_mid],
                T_vec[end], q_grid[idx, end])
    end
    println()
    total_mc_chains = 2 * sum(n_disorder_vec) * n_T  # 2 replicas
    total_mc_steps = total_mc_chains * (N_EQ + N_SAMP)
    @printf("Total GPU time: %.1f s (eq: %.1f + samp: %.1f)\n",
            t_total_eq+t_total_samp, t_total_eq, t_total_samp)
    @printf("Total MC steps: %.2e  (2 replicas × %d disorder avg)\n",
            Float64(total_mc_steps), sum(n_disorder_vec))
    println("=" ^ 70)
    println("Protocol summary:")
    println("  Step size: σ(N,T) = 2.4·T/√N (v6)")
    println("  Equilibration: $N_EQ steps (transient discarded)")
    println("  Sampling: $N_SAMP steps (φ and q measured)")
    println("  q_EA = ⟨x_a·x_b/N⟩ (two-replica cross-overlap)")
    println("  Patterns shared between replicas → ~50% less GPU memory than v6")
    println("Interpretation:")
    println("  φ ≈ 1: Basin stable → successful retrieval")
    println("  φ < 0.75: Basin unstable → retrieval fails")
    println("  Hard wall at φ_c ≈ 0.707 prevents escape from basin")
    println("  q_EA ≈ 1: Frozen (glassy) phase")
    println("  q_EA ≈ 0: Ergodic (liquid) phase")
    println("=" ^ 70)
end

main()
