#=
GPU-Accelerated LSR Alpha Sweep – All-T Parallel (v5)
- Same MC protocol as LSE (all T in parallel, single equilibration pass)
- LSR energy with b = 2 + √2 (Epanechnikov kernel)
- P range: 500–20,000 (linear in α)
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

const alpha_vec = collect(F(0.01):F(0.01):F(0.55))  # 55 values
const T_vec     = F.(10 .^ range(-2, log10(2.5), length=50))  # log-spaced: 0.01 → 2.5
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)
const N_TRIALS  = 256
const N_EQ      = 50000
const N_SAMP    = 5000
const MIN_PAT   = 500
const MAX_PAT   = 20000
const n_chains  = n_T * N_TRIALS  # 12800

# Adaptive pattern count (linear in α)
function n_patterns(alpha)
    slope = (MAX_PAT - MIN_PAT) / (alpha_vec[end] - alpha_vec[1])
    return MIN_PAT + slope * (alpha - alpha_vec[1])
end

# Adaptive step size
adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── LSR Energy (batched over trials) ────────────────
const INF_ENERGY = F(1e30)

function compute_energy_lsr!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F)
    Nb = Nf / b_lsr
    # Batched gemm: overlap[p,j,t] = Σ_n patterns[n,p,t] * x[n,j,t]
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)

    @. overlap = max(zero(F), one(F) - b_lsr + b_lsr * overlap / Nf)
    s = sum(overlap, dims=1)
    E .= vec(@. ifelse(s > zero(F), -Nb * log(s), INF_ENERGY))
    return nothing
end

# ──────────────── MC Step for one α ────────────────
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
    compute_energy_lsr!(Ep, xp, pat, ov, Nf)

    # Metropolis accept/reject (reject if basin escape)
    CUDA.rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-(β * (Ep - E))))
    a3 = reshape(acc, 1, n_T, N_TRIALS)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 70)
    println("LSR Alpha Sweep – GPU v5 (All-T Parallel, b=$(round(b_lsr, digits=3)))")
    println("=" ^ 70)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # Compute N(α), P(α) for all α
    Ns = Int[]; Ps = Int[]
    for α in alpha_vec
        Pt = n_patterns(α)
        N  = max(round(Int, log(Pt) / α), 2)
        push!(Ns, N); push!(Ps, round(Int, Pt))
    end
    @printf("N range: %d – %d,  P range: %d – %d\n", extrema(Ns)..., extrema(Ps)...)
    @printf("Grid: %d α × %d T,  trials: %d,  MC: %d eq + %d samp\n\n",
            n_alpha, n_T, N_TRIALS, N_EQ, N_SAMP)

    # ── Allocate GPU data for ALL α simultaneously ──
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

        # Generate normalised patterns on CPU (reproducible seed per α)
        Random.seed!(42 + i)
        p_cpu = randn(F, N, P, N_TRIALS)
        for t in 1:N_TRIALS, j in 1:P
            c = @view p_cpu[:, j, t]
            c .*= sqrt(Nf) / norm(c)
        end

        pats_g[i]  = CuArray(p_cpu)
        tgts_g[i]  = CuArray(p_cpu[:, 1:1, :])       # [N × 1 × N_TRIALS]
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
    GC.gc()  # free CPU-side pattern arrays

    β_gpu = CuVector{F}(repeat(F.(1 ./ T_vec), N_TRIALS))
    ra    = CUDA.zeros(F, n_chains)

    @printf("GPU allocated: %.2f GB  (free: %.2f GB)\n\n",
            mem/1e9, CUDA.available_memory()/1e9)

    # ── Initialize states near targets ──
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

    # ── Equilibration ──
    println("Equilibration ($N_EQ steps × $n_alpha α values)...")
    t0 = time()
    prog = Progress(N_EQ, desc="Equilibration: ")
    for step in 1:N_EQ
        for i in 1:n_alpha
            mc_step!(xs_g[i], xps_g[i], Es_g[i], Eps_g[i],
                     pats_g[i], ovs_g[i], β_gpu, ra,
                     F(Ns[i]), ssvec[i])
        end
        next!(prog)
    end
    finish!(prog)
    CUDA.synchronize()
    t_eq = time() - t0
    @printf("Equilibration: %.1f s (%.2f ms/step)\n\n", t_eq, 1000*t_eq/N_EQ)

    # ── Sampling ──
    println("Sampling ($N_SAMP steps × $n_alpha α values)...")
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
            # Measure alignment: φ = (target · x) / N
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
        phi_avg = Array(phis_g[i]) ./ N_SAMP         # [n_chains]
        phi_mat = reshape(phi_avg, n_T, N_TRIALS)     # [n_T × N_TRIALS]
        phi_grid[i, :] = vec(mean(phi_mat, dims=2))   # average over trials
    end

    # ── Save CSV (identical format to LSE version) ──
    csv_file = "lsr_longeq.csv"
    open(csv_file, "w") do f
        write(f, "alpha")
        for T in T_vec
            write(f, @sprintf(",T%.4f", T))
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

    # Sample output
    println("Sample data:")
    j_mid = n_T ÷ 2
    for idx in [1, n_alpha÷2, n_alpha]
        @printf("  α=%.2f (N=%d, P=%d): φ(T=%.3f)=%.4f, φ(T=%.3f)=%.4f, φ(T=%.3f)=%.4f\n",
                alpha_vec[idx], Ns[idx], Ps[idx],
                T_vec[1], phi_grid[idx, 1],
                T_vec[j_mid], phi_grid[idx, j_mid],
                T_vec[end], phi_grid[idx, end])
    end
    println()
    @printf("Total GPU time: %.1f s (eq: %.1f + samp: %.1f)\n", t_eq+t_samp, t_eq, t_samp)
    println("=" ^ 70)
end

main()
