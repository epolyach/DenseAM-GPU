#=
GPU-Accelerated LSR Alpha Sweep – T-loop version
- Loops over T values, each gets its own N_EQ(T) steps
- No wasted compute on frozen chains
- Cleaner structure, lower error probability
- Output: lsr_longeq.csv
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

const F = Float32

# ──────────────── Parameters ────────────────
const b_lsr     = F(2 + sqrt(2))   # ≈ 3.414

const alpha_vec = collect(F(0.01):F(0.01):F(0.55))  # 55 values (production)
const T_vec     = collect(F(0.05):F(0.05):F(2.50))  # 50 values (production)

# const alpha_vec = collect(F(0.2):F(0.02):F(0.54))  
# const T_vec     = collect(F(0.05):F(0.05):F(1.50)) 

const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

const N_TRIALS  = 256  # 64
const N_SAMP    = 5000 # 500
const MIN_PAT   = 500 # 500
const MAX_PAT   = 20000 # 20000
const n_chains  = n_T * N_TRIALS  # 12800

# ──────────────── T-dependent equilibration ────────────────
const N_EQ_BASE = 5000
const N_EQ_CAP  = 300000
const EQ_COEFF  = F(0.15)

function n_eq_for_T(T::Real)
    T_max = maximum(T_vec)
    n_eq = N_EQ_BASE * exp(EQ_COEFF * (1/T - 1/T_max))
    return min(N_EQ_CAP, round(Int, n_eq))
end

const N_EQ_vec = [n_eq_for_T(T) for T in T_vec]
const TOTAL_EQ_STEPS = sum(N_EQ_vec)

# ──────────────── Adaptive functions ────────────────
function n_patterns(alpha)
    slope = (MAX_PAT - MIN_PAT) / (alpha_vec[end] - alpha_vec[1])
    return MIN_PAT + slope * (alpha - alpha_vec[1])
end

adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── LSR Energy ────────────────
const INF_ENERGY = F(1e30)

function compute_energy_lsr!(E::CuVector{F}, x::CuArray{F,2},
                              patterns::CuArray{F,3}, overlap::CuArray{F,2},
                              Nf::F)
    # x: [N, N_TRIALS], patterns: [N, P, N_TRIALS], overlap: [P, N_TRIALS]
    Nb = Nf / b_lsr
    nTrials = size(x, 2)

    # Compute overlap for each trial: overlap[:, t] = patterns[:,:,t]' * x[:,t]
    for t in 1:nTrials
        CUDA.CUBLAS.gemv!('T', one(F), view(patterns, :, :, t), view(x, :, t), zero(F), view(overlap, :, t))
    end

    @. overlap = max(zero(F), one(F) - b_lsr + b_lsr * overlap / Nf)
    s = sum(overlap, dims=1)  # [1, N_TRIALS]
    E .= vec(@. ifelse(s > zero(F), -Nb * log(s), INF_ENERGY))
    return nothing
end

# Batched version using gemm_strided_batched
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
    # x, xp: [N, 1, N_TRIALS]
    nTrials = size(x, 3)

    # Propose on sphere
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    # Proposed energy
    compute_energy_lsr_batched!(Ep, xp, pat, ov, Nf)

    # Metropolis accept/reject
    CUDA.rand!(ra)
    acc = @. ra < exp(-β * (Ep - E))
    a3 = reshape(acc, 1, 1, nTrials)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 70)
    println("LSR Alpha Sweep – GPU v2 (T-loop, b=$(round(b_lsr, digits=3)))")
    println("=" ^ 70)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # Compute N(α), P(α) for all α
    Ns = Int[]; Ps = Int[]
    for α in alpha_vec
        Pt = n_patterns(α)
        N = max(round(Int, log(Pt) / α), 2)
        push!(Ns, N); push!(Ps, round(Int, Pt))
    end
    @printf("N range: %d – %d,  P range: %d – %d\n", extrema(Ns)..., extrema(Ps)...)
    @printf("Grid: %d α × %d T,  trials: %d\n", n_alpha, n_T, N_TRIALS)
    @printf("T-dependent equilibration: N_EQ(T) = %d × exp(%.2f × (1/T - 1/T_max)), cap=%d\n",
            N_EQ_BASE, EQ_COEFF, N_EQ_CAP)
    @printf("  T=%.2f: N_EQ=%d,  T=%.2f: N_EQ=%d,  T=%.2f: N_EQ=%d\n",
            T_vec[end], N_EQ_vec[end], T_vec[n_T÷2], N_EQ_vec[n_T÷2], T_vec[1], N_EQ_vec[1])
    @printf("  Total equilibration steps: %d (across all T)\n\n", TOTAL_EQ_STEPS)

    # ── Allocate GPU data ──
    # For each α: patterns [N, P, N_TRIALS], target [N, 1, N_TRIALS]
    # State arrays are per-T: x [N, 1, N_TRIALS] for current T
    println("Allocating GPU memory...")

    pats_g = Vector{CuArray{F,3}}(undef, n_alpha)
    tgts_g = Vector{CuArray{F,3}}(undef, n_alpha)
    ssvec  = Vector{F}(undef, n_alpha)

    # Full state storage: xs_full[i][j] = state for α[i], T[j]
    # Shape: [N, 1, N_TRIALS] for each (α, T) pair
    xs_full = [[CUDA.zeros(F, Ns[i], 1, N_TRIALS) for j in 1:n_T] for i in 1:n_alpha]

    # Working arrays (reused across T)
    xp_g   = Vector{CuArray{F,3}}(undef, n_alpha)
    ov_g   = Vector{CuArray{F,3}}(undef, n_alpha)
    E_g    = Vector{CuVector{F}}(undef, n_alpha)
    Ep_g   = Vector{CuVector{F}}(undef, n_alpha)

    mem = 0
    for i in 1:n_alpha
        N = Ns[i]; P = Ps[i]; Nf = F(N)

        Random.seed!(42 + i)
        p_cpu = randn(F, N, P, N_TRIALS)
        for t in 1:N_TRIALS, j in 1:P
            c = @view p_cpu[:, j, t]
            c .*= sqrt(Nf) / norm(c)
        end

        pats_g[i] = CuArray(p_cpu)
        tgts_g[i] = CuArray(p_cpu[:, 1:1, :])  # [N, 1, N_TRIALS]
        xp_g[i]   = CUDA.zeros(F, N, 1, N_TRIALS)
        ov_g[i]   = CUDA.zeros(F, P, 1, N_TRIALS)
        E_g[i]    = CUDA.zeros(F, N_TRIALS)
        Ep_g[i]   = CUDA.zeros(F, N_TRIALS)
        ssvec[i]  = adaptive_ss(N)

        mem += (N*P*N_TRIALS + N*N_TRIALS + N*n_T*N_TRIALS + N*N_TRIALS + P*N_TRIALS + 2*N_TRIALS) * sizeof(F)
    end
    GC.gc()

    ra = CUDA.zeros(F, N_TRIALS)

    @printf("GPU allocated: %.2f GB  (free: %.2f GB)\n\n", mem/1e9, CUDA.available_memory()/1e9)

    # ── Initialize all states near targets ──
    println("Initializing states for all (α, T)...")
    for i in 1:n_alpha
        Nf = F(Ns[i])
        for j in 1:n_T
            xs_full[i][j] .= tgts_g[i] .+ F(0.05) .* CUDA.randn(F, Ns[i], 1, N_TRIALS)
            nrm = sqrt.(sum(xs_full[i][j] .^ 2, dims=1))
            xs_full[i][j] .= sqrt(Nf) .* xs_full[i][j] ./ nrm
        end
    end
    CUDA.synchronize()
    println("Done.\n")

    # ── Equilibration: loop over T, each gets N_EQ(T) steps ──
    println("Equilibration (T-loop)...")
    t0 = time()

    prog = Progress(TOTAL_EQ_STEPS, desc="Equilibration: ")
    steps_done = 0

    for j in 1:n_T
        T = T_vec[j]
        β = F(1 / T)
        n_eq = N_EQ_vec[j]

        for step in 1:n_eq
            for i in 1:n_alpha
                mc_step_single_T!(xs_full[i][j], xp_g[i], E_g[i], Ep_g[i],
                                  pats_g[i], ov_g[i], β, ra,
                                  F(Ns[i]), ssvec[i])
            end
            steps_done += 1
            next!(prog)
        end
    end
    finish!(prog)
    CUDA.synchronize()
    t_eq = time() - t0
    @printf("Equilibration: %.1f s (%.2f ms/step avg)\n\n", t_eq, 1000*t_eq/TOTAL_EQ_STEPS)

    # ── Sampling: all (α, T) get same N_SAMP ──
    println("Sampling ($N_SAMP steps × $n_alpha α × $n_T T)...")

    # Accumulate φ for each (α, T)
    phi_acc = [[CUDA.zeros(F, N_TRIALS) for j in 1:n_T] for i in 1:n_alpha]

    t0 = time()
    prog = Progress(N_SAMP, desc="Sampling: ")

    for step in 1:N_SAMP
        for j in 1:n_T
            T = T_vec[j]
            β = F(1 / T)

            for i in 1:n_alpha
                Nf = F(Ns[i])
                mc_step_single_T!(xs_full[i][j], xp_g[i], E_g[i], Ep_g[i],
                                  pats_g[i], ov_g[i], β, ra,
                                  Nf, ssvec[i])
                # Measure φ = (target · x) / N
                phi_acc[i][j] .+= vec(sum(tgts_g[i] .* xs_full[i][j], dims=1)) ./ Nf
            end
        end
        next!(prog)
    end
    finish!(prog)
    CUDA.synchronize()
    t_samp = time() - t0
    @printf("Sampling: %.1f s (%.2f ms/step)\n\n", t_samp, 1000*t_samp/N_SAMP)

    # ── Stream CSV results (write header + each α as it completes) ──
    csv_file = "lsr_longeq.csv"
    csv_handle = open(csv_file, "w")
    
    # Write header
    write(csv_handle, "alpha")
    for T in T_vec
        write(csv_handle, @sprintf(",T%.2f", T))
    end
    write(csv_handle, "\n")
    flush(csv_handle)  # Ensure header is written immediately
    
    # Collect and stream results α by α
    phi_grid = zeros(Float64, n_alpha, n_T)
    println("Streaming results to $csv_file as sampling completes...")
    for i in 1:n_alpha
        for j in 1:n_T
            phi_avg = Array(phi_acc[i][j]) ./ N_SAMP  # [N_TRIALS]
            phi_grid[i, j] = mean(phi_avg)
        end
        # Write this α's row immediately
        write(csv_handle, @sprintf("%.2f", alpha_vec[i]))
        for j in 1:n_T
            write(csv_handle, @sprintf(",%.4f", phi_grid[i, j]))
        end
        write(csv_handle, "\n")
        flush(csv_handle)  # Force write to disk
    end
    close(csv_handle)
    println("CSV saved: $csv_file (streamed as results were computed)")
    println()

    # Sample output
    println("Sample data:")
    for idx in [1, n_alpha÷2, n_alpha]
        j_1 = findfirst(t -> t ≈ F(1.0), T_vec)
        j_1 = isnothing(j_1) ? n_T÷2 : j_1
        @printf("  α=%.2f (N=%d, P=%d): φ(T=%.2f)=%.4f, φ(T=%.2f)=%.4f, φ(T=%.2f)=%.4f\n",
                alpha_vec[idx], Ns[idx], Ps[idx],
                T_vec[1], phi_grid[idx, 1],
                T_vec[j_1], phi_grid[idx, j_1],
                T_vec[end], phi_grid[idx, end])
    end
    println()
    @printf("Total GPU time: %.1f s (eq: %.1f + samp: %.1f)\n", t_eq+t_samp, t_eq, t_samp)
    @printf("Equilibration: %d total steps (T-dependent)\n", TOTAL_EQ_STEPS)
    @printf("b = %.4f (Epanechnikov kernel)\n", b_lsr)
    println("=" ^ 70)
end

main()
