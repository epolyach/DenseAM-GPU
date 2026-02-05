#=
GPU-Accelerated LSE Alpha Sweep – Dual Initialization + Parallel Tempering
- 256 trials split: 128 warm (near target) + 128 cold (random on sphere)
- Parallel tempering with even/odd replica exchange every MC step
- Measures φ separately for each group → metastability diagnostic
- Output: lse_dual_warm.csv, lse_dual_cold.csv
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

const F = Float32

# ──────────────── Parameters ────────────────
const betanet = F(1.0)

const alpha_vec = collect(F(0.01):F(0.01):F(0.55))  # 55 values
const T_vec     = collect(F(0.05):F(0.05):F(2.50))   # 50 values
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)
const N_TRIALS  = 256
const N_WARM    = 128   # trials 1–128: near target
const N_COLD    = 128   # trials 129–256: random
const N_EQ      = 50000
const N_SAMP    = 5000
const MIN_PAT   = 200
const MAX_PAT   = 5000
const n_chains  = n_T * N_TRIALS  # 12800

# Adaptive pattern count (linear in α)
function n_patterns(alpha)
    slope = (MAX_PAT - MIN_PAT) / (alpha_vec[end] - alpha_vec[1])
    return MIN_PAT + slope * (alpha - alpha_vec[1])
end

# Adaptive step size
adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── Parallel tempering: precompute Δβ ────────────────
const j_even_lo = collect(1:2:(n_T-1))
const j_even_hi = collect(2:2:n_T)
const n_even    = length(j_even_lo)
const Δβ_even_cpu = F[F(1/T_vec[j_even_lo[k]] - 1/T_vec[j_even_hi[k]]) for k in 1:n_even]

const j_odd_lo  = collect(2:2:(n_T-1))
const j_odd_hi  = collect(3:2:n_T)
const n_odd     = length(j_odd_lo)
const Δβ_odd_cpu = F[F(1/T_vec[j_odd_lo[k]] - 1/T_vec[j_odd_hi[k]]) for k in 1:n_odd]

# ──────────────── LSE Energy (batched over trials) ────────────────
function compute_energy_lse!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = -betanet * (Nf - overlap)
    m = maximum(overlap, dims=1)
    @. overlap = exp(overlap - m)
    s = sum(overlap, dims=1)
    E .= vec(@. -(m + log(s)) / betanet)
    return nothing
end

# ──────────────── MC Step for one α ────────────────
function mc_step!(x::CuArray{F,3}, xp::CuArray{F,3},
                  E::CuVector{F}, Ep::CuVector{F},
                  pat::CuArray{F,3}, ov::CuArray{F,3},
                  β::CuVector{F}, ra::CuVector{F},
                  Nf::F, ss::F)
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lse!(Ep, xp, pat, ov, Nf)

    CUDA.rand!(ra)
    acc = @. ra < exp(-(β * (Ep - E)))
    a3 = reshape(acc, 1, n_T, N_TRIALS)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Replica exchange swap ────────────────
function replica_swap!(x::CuArray{F,3}, E::CuVector{F},
                       Δβ_gpu::CuVector{F},
                       j_lo::Vector{Int}, j_hi::Vector{Int},
                       np::Int)
    E_mat = reshape(E, n_T, N_TRIALS)
    ΔE = E_mat[j_lo, :] .- E_mat[j_hi, :]
    log_p = Δβ_gpu .* ΔE
    r = CUDA.rand(F, np, N_TRIALS)
    acc = r .< exp.(log_p)

    E1 = E_mat[j_lo, :]; E2 = E_mat[j_hi, :]
    E_mat[j_lo, :] .= ifelse.(acc, E2, E1)
    E_mat[j_hi, :] .= ifelse.(acc, E1, E2)

    acc3 = reshape(acc, 1, np, N_TRIALS)
    x1 = x[:, j_lo, :]; x2 = x[:, j_hi, :]
    x[:, j_lo, :] .= ifelse.(acc3, x2, x1)
    x[:, j_hi, :] .= ifelse.(acc3, x1, x2)
    return nothing
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 70)
    println("LSE Alpha Sweep – GPU (Dual Init + PT, N_EQ=$N_EQ)")
    println("  Warm trials: 1–$N_WARM (near target)")
    println("  Cold trials: $(N_WARM+1)–$N_TRIALS (random on sphere)")
    println("=" ^ 70)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    Δβ_even_gpu = CuVector{F}(Δβ_even_cpu)
    Δβ_odd_gpu  = CuVector{F}(Δβ_odd_cpu)

    # Compute N(α), P(α) for all α
    Ns = Int[]; Ps = Int[]
    for α in alpha_vec
        Pt = n_patterns(α)
        N  = max(round(Int, log(Pt) / α), 2)
        push!(Ns, N); push!(Ps, round(Int, Pt))
    end
    @printf("N range: %d – %d,  P range: %d – %d\n", extrema(Ns)..., extrema(Ps)...)
    @printf("Grid: %d α × %d T,  trials: %d (%d warm + %d cold)\n",
            n_alpha, n_T, N_TRIALS, N_WARM, N_COLD)
    @printf("MC: %d eq + %d samp,  PT: even/odd swap every step\n\n", N_EQ, N_SAMP)

    # ── Allocate GPU data ──
    println("Allocating GPU memory...")
    pats_g  = Vector{CuArray{F,3}}(undef, n_alpha)
    tgts_g  = Vector{CuArray{F,3}}(undef, n_alpha)
    xs_g    = Vector{CuArray{F,3}}(undef, n_alpha)
    xps_g   = Vector{CuArray{F,3}}(undef, n_alpha)
    ovs_g   = Vector{CuArray{F,3}}(undef, n_alpha)
    Es_g    = Vector{CuVector{F}}(undef, n_alpha)
    Eps_g   = Vector{CuVector{F}}(undef, n_alpha)
    phis_warm_g = Vector{CuVector{F}}(undef, n_alpha)
    phis_cold_g = Vector{CuVector{F}}(undef, n_alpha)
    ssvec   = Vector{F}(undef, n_alpha)

    mem = 0
    for i in 1:n_alpha
        N = Ns[i]; P = Ps[i]; Nf = F(N)

        Random.seed!(42 + i)
        p_cpu = randn(F, N, P, N_TRIALS)
        for t in 1:N_TRIALS, j in 1:P
            c = @view p_cpu[:, j, t]
            c .*= sqrt(Nf) / norm(c)
        end

        pats_g[i]  = CuArray(p_cpu)
        tgts_g[i]  = CuArray(p_cpu[:, 1:1, :])
        xs_g[i]    = CUDA.zeros(F, N, n_T, N_TRIALS)
        xps_g[i]   = CUDA.zeros(F, N, n_T, N_TRIALS)
        ovs_g[i]   = CUDA.zeros(F, P, n_T, N_TRIALS)
        Es_g[i]    = CUDA.zeros(F, n_chains)
        Eps_g[i]   = CUDA.zeros(F, n_chains)
        phis_warm_g[i] = CUDA.zeros(F, n_T * N_WARM)
        phis_cold_g[i] = CUDA.zeros(F, n_T * N_COLD)
        ssvec[i]   = adaptive_ss(N)

        mem += (N*P*N_TRIALS + N*N_TRIALS + 2*N*n_T*N_TRIALS +
                P*n_T*N_TRIALS + 3*n_chains) * sizeof(F)
    end
    GC.gc()

    β_gpu = CuVector{F}(repeat(F.(1 ./ T_vec), N_TRIALS))
    ra    = CUDA.zeros(F, n_chains)

    @printf("GPU allocated: %.2f GB  (free: %.2f GB)\n\n",
            mem/1e9, CUDA.available_memory()/1e9)

    # ── Dual initialization ──
    println("Initializing states (dual: warm + cold)...")
    for i in 1:n_alpha
        Nf = F(Ns[i])
        # Warm trials (1–N_WARM): near target with 5% noise
        xs_g[i][:, :, 1:N_WARM] .= tgts_g[i][:, :, 1:N_WARM] .+
            F(0.05) .* CUDA.randn(F, Ns[i], n_T, N_WARM)
        # Cold trials (N_WARM+1–N_TRIALS): random on sphere
        xs_g[i][:, :, (N_WARM+1):N_TRIALS] .= CUDA.randn(F, Ns[i], n_T, N_COLD)
        # Normalize all to √N-sphere
        nrm = sqrt.(sum(xs_g[i] .^ 2, dims=1))
        xs_g[i] .= sqrt(Nf) .* xs_g[i] ./ nrm
        compute_energy_lse!(Es_g[i], xs_g[i], pats_g[i], ovs_g[i], Nf)
    end
    CUDA.synchronize()
    println("Done.\n")

    # ── Equilibration with PT ──
    println("Equilibration ($N_EQ steps × $n_alpha α values, with PT swaps)...")
    t0 = time()
    prog = Progress(N_EQ, desc="Equilibration: ")
    for step in 1:N_EQ
        for i in 1:n_alpha
            mc_step!(xs_g[i], xps_g[i], Es_g[i], Eps_g[i],
                     pats_g[i], ovs_g[i], β_gpu, ra,
                     F(Ns[i]), ssvec[i])
            if iseven(step)
                replica_swap!(xs_g[i], Es_g[i], Δβ_even_gpu,
                              j_even_lo, j_even_hi, n_even)
            else
                replica_swap!(xs_g[i], Es_g[i], Δβ_odd_gpu,
                              j_odd_lo, j_odd_hi, n_odd)
            end
        end
        next!(prog)
    end
    finish!(prog)
    CUDA.synchronize()
    t_eq = time() - t0
    @printf("Equilibration: %.1f s (%.2f ms/step)\n\n", t_eq, 1000*t_eq/N_EQ)

    # ── Sampling with PT (separate warm/cold accumulators) ──
    println("Sampling ($N_SAMP steps × $n_alpha α values, with PT swaps)...")
    for i in 1:n_alpha
        phis_warm_g[i] .= zero(F)
        phis_cold_g[i] .= zero(F)
    end

    t0 = time()
    prog = Progress(N_SAMP, desc="Sampling: ")
    for step in 1:N_SAMP
        for i in 1:n_alpha
            Nf = F(Ns[i])
            mc_step!(xs_g[i], xps_g[i], Es_g[i], Eps_g[i],
                     pats_g[i], ovs_g[i], β_gpu, ra,
                     Nf, ssvec[i])
            if iseven(step)
                replica_swap!(xs_g[i], Es_g[i], Δβ_even_gpu,
                              j_even_lo, j_even_hi, n_even)
            else
                replica_swap!(xs_g[i], Es_g[i], Δβ_odd_gpu,
                              j_odd_lo, j_odd_hi, n_odd)
            end

            # Measure alignment, split by group
            phi_all = vec(sum(tgts_g[i] .* xs_g[i], dims=1)) ./ Nf
            phis_warm_g[i] .+= phi_all[1:(n_T*N_WARM)]
            phis_cold_g[i] .+= phi_all[(n_T*N_WARM+1):end]
        end
        next!(prog)
    end
    finish!(prog)
    CUDA.synchronize()
    t_samp = time() - t0
    @printf("Sampling: %.1f s (%.2f ms/step)\n\n", t_samp, 1000*t_samp/N_SAMP)

    # ── Collect results ──
    println("Collecting results...")
    phi_warm_grid = zeros(Float64, n_alpha, n_T)
    phi_cold_grid = zeros(Float64, n_alpha, n_T)
    for i in 1:n_alpha
        pw = Array(phis_warm_g[i]) ./ N_SAMP
        phi_warm_grid[i, :] = vec(mean(reshape(pw, n_T, N_WARM), dims=2))
        pc = Array(phis_cold_g[i]) ./ N_SAMP
        phi_cold_grid[i, :] = vec(mean(reshape(pc, n_T, N_COLD), dims=2))
    end

    # ── Save CSVs ──
    for (grid, label) in [(phi_warm_grid, "warm"), (phi_cold_grid, "cold")]
        csv_file = "lse_dual_$(label).csv"
        open(csv_file, "w") do f
            write(f, "alpha")
            for T in T_vec
                write(f, @sprintf(",T%.2f", T))
            end
            write(f, "\n")
            for i in 1:n_alpha
                write(f, @sprintf("%.2f", alpha_vec[i]))
                for j in 1:n_T
                    write(f, @sprintf(",%.4f", grid[i, j]))
                end
                write(f, "\n")
            end
        end
        println("CSV saved: $csv_file")
    end
    println()

    # Sample output
    println("Sample data (warm | cold):")
    for idx in [1, n_alpha÷2, n_alpha]
        j_1 = findfirst(t -> t ≈ F(1.0), T_vec)
        @printf("  α=%.2f (N=%d, P=%d):\n", alpha_vec[idx], Ns[idx], Ps[idx])
        @printf("    warm: φ(T=0.05)=%.4f, φ(T=1.00)=%.4f, φ(T=2.50)=%.4f\n",
                phi_warm_grid[idx, 1], phi_warm_grid[idx, j_1], phi_warm_grid[idx, end])
        @printf("    cold: φ(T=0.05)=%.4f, φ(T=1.00)=%.4f, φ(T=2.50)=%.4f\n",
                phi_cold_grid[idx, 1], phi_cold_grid[idx, j_1], phi_cold_grid[idx, end])
    end
    println()
    @printf("Total GPU time: %.1f s (eq: %.1f + samp: %.1f)\n", t_eq+t_samp, t_eq, t_samp)
    println("Dual init: $N_WARM warm + $N_COLD cold, PT: even/odd swap every step")
    println("=" ^ 70)
end

main()
