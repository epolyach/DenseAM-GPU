#=
GPU-Accelerated LSR Alpha Sweep with Parallel Tempering (Replica Exchange)
- Same adaptive-N approach as longeq (P 200–5000, 256 trials)
- N_EQ = 50,000, N_SAMP = 5,000 (same as longeq for comparison)
- Epanechnikov kernel with b = 2 + √2
- After each MC step, attempt replica exchange swaps between adjacent
  temperature chains within each trial (even/odd alternating)
- Swap acceptance: min(1, exp((β_i − β_{i+1})(E_i − E_{i+1})))
- Output: lsr_pt.csv
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

const alpha_vec = collect(F(0.01):F(0.01):F(0.55))  # 55 values
const T_vec     = collect(F(0.05):F(0.05):F(2.50))   # 50 values
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)
const N_TRIALS  = 256
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
# Even pairs: (1,2), (3,4), ..., (49,50)  → 25 pairs
const j_even_lo = collect(1:2:(n_T-1))
const j_even_hi = collect(2:2:n_T)
const n_even    = length(j_even_lo)
const Δβ_even_cpu = F[F(1/T_vec[j_even_lo[k]] - 1/T_vec[j_even_hi[k]]) for k in 1:n_even]

# Odd pairs: (2,3), (4,5), ..., (48,49)  → 24 pairs
const j_odd_lo  = collect(2:2:(n_T-1))
const j_odd_hi  = collect(3:2:n_T)
const n_odd     = length(j_odd_lo)
const Δβ_odd_cpu = F[F(1/T_vec[j_odd_lo[k]] - 1/T_vec[j_odd_hi[k]]) for k in 1:n_odd]

# ──────────────── LSR Energy (batched over trials) ────────────────
function compute_energy_lsr!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F)
    Nb = Nf / b_lsr

    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)

    # LSR: E = -(N/b) * log(Σ_p max(0, 1 - b + b*overlap/N))
    @. overlap = max(zero(F), one(F) - b_lsr + b_lsr * overlap / Nf)
    s = sum(overlap, dims=1)
    @. s = max(s, F(1e-10))
    E .= vec(@. -Nb * log(s))
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

    compute_energy_lsr!(Ep, xp, pat, ov, Nf)

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
    println("LSR Alpha Sweep – GPU (Parallel Tempering, N_EQ=$N_EQ, b=$(round(b_lsr, digits=3)))")
    println("=" ^ 70)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    Δβ_even_gpu = CuVector{F}(Δβ_even_cpu)
    Δβ_odd_gpu  = CuVector{F}(Δβ_odd_cpu)

    Ns = Int[]; Ps = Int[]
    for α in alpha_vec
        Pt = n_patterns(α)
        N  = max(round(Int, log(Pt) / α), 2)
        push!(Ns, N); push!(Ps, round(Int, Pt))
    end
    @printf("N range: %d – %d,  P range: %d – %d\n", extrema(Ns)..., extrema(Ps)...)
    @printf("Grid: %d α × %d T,  trials: %d,  MC: %d eq + %d samp\n",
            n_alpha, n_T, N_TRIALS, N_EQ, N_SAMP)
    println("Parallel tempering: even/odd swap every MC step\n")

    println("Allocating GPU memory...")
    pats_g  = Vector{CuArray{F,3}}(undef, n_alpha)
    tgts_g  = Vector{CuArray{F,3}}(undef, n_alpha)
    xs_g    = Vector{CuArray{F,3}}(undef, n_alpha)
    xps_g   = Vector{CuArray{F,3}}(undef, n_alpha)
    ovs_g   = Vector{CuArray{F,3}}(undef, n_alpha)
    Es_g    = Vector{CuVector{F}}(undef, n_alpha)
    Eps_g   = Vector{CuVector{F}}(undef, n_alpha)
    phis_g  = Vector{CuVector{F}}(undef, n_alpha)
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
        phis_g[i]  = CUDA.zeros(F, n_chains)
        ssvec[i]   = adaptive_ss(N)

        mem += (N*P*N_TRIALS + N*N_TRIALS + 2*N*n_T*N_TRIALS +
                P*n_T*N_TRIALS + 3*n_chains) * sizeof(F)
    end
    GC.gc()

    β_gpu = CuVector{F}(repeat(F.(1 ./ T_vec), N_TRIALS))
    ra    = CUDA.zeros(F, n_chains)

    @printf("GPU allocated: %.2f GB  (free: %.2f GB)\n\n",
            mem/1e9, CUDA.available_memory()/1e9)

    println("Initializing states...")
    for i in 1:n_alpha
        Nf = F(Ns[i])
        xs_g[i] .= tgts_g[i] .+ F(0.05) .* CUDA.randn(F, Ns[i], n_T, N_TRIALS)
        nrm = sqrt.(sum(xs_g[i] .^ 2, dims=1))
        xs_g[i] .= sqrt(Nf) .* xs_g[i] ./ nrm
        compute_energy_lsr!(Es_g[i], xs_g[i], pats_g[i], ovs_g[i], Nf)
    end
    CUDA.synchronize()
    println("Done.\n")

    # ── Equilibration with parallel tempering ──
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

    # ── Sampling with parallel tempering ──
    println("Sampling ($N_SAMP steps × $n_alpha α values, with PT swaps)...")
    for i in 1:n_alpha
        phis_g[i] .= zero(F)
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

            phis_g[i] .+= vec(sum(tgts_g[i] .* xs_g[i], dims=1)) ./ Nf
        end
        next!(prog)
    end
    finish!(prog)
    CUDA.synchronize()
    t_samp = time() - t0
    @printf("Sampling: %.1f s (%.2f ms/step)\n\n", t_samp, 1000*t_samp/N_SAMP)

    # ── Collect results ──
    println("Collecting results...")
    phi_grid = zeros(Float64, n_alpha, n_T)
    for i in 1:n_alpha
        phi_avg = Array(phis_g[i]) ./ N_SAMP
        phi_mat = reshape(phi_avg, n_T, N_TRIALS)
        phi_grid[i, :] = vec(mean(phi_mat, dims=2))
    end

    # ── Save CSV ──
    csv_file = "lsr_pt.csv"
    open(csv_file, "w") do f
        write(f, "alpha")
        for T in T_vec
            write(f, @sprintf(",T%.2f", T))
        end
        write(f, "\n")
        for i in 1:n_alpha
            write(f, @sprintf("%.2f", alpha_vec[i]))
            for j in 1:n_T
                write(f, @sprintf(",%.4f", phi_grid[i, j]))
            end
            write(f, "\n")
        end
    end
    println("CSV saved: $csv_file")
    println()

    println("Sample data:")
    for idx in [1, n_alpha÷2, n_alpha]
        j_1 = findfirst(t -> t ≈ F(1.0), T_vec)
        @printf("  α=%.2f (N=%d, P=%d): φ(T=0.05)=%.4f, φ(T=1.00)=%.4f, φ(T=2.50)=%.4f\n",
                alpha_vec[idx], Ns[idx], Ps[idx],
                phi_grid[idx, 1], phi_grid[idx, j_1], phi_grid[idx, end])
    end
    println()
    @printf("Total GPU time: %.1f s (eq: %.1f + samp: %.1f)\n", t_eq+t_samp, t_eq, t_samp)
    @printf("b = %.4f (Epanechnikov kernel)\n", b_lsr)
    println("Parallel tempering: even/odd replica exchange every MC step")
    println("=" ^ 70)
end

main()
