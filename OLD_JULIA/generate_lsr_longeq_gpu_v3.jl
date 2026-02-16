#=
GPU-Accelerated LSR Alpha Sweep – Heating protocol (v3)
- Propagates equilibrated states from low T → high T
- Removes metastable "blue bay" by letting thermal fluctuations
  naturally destabilize retrieval as T crosses T_c
- Output: lsr_heating.csv
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

const alpha_vec = collect(F(0.2):F(0.02):F(0.54))
const T_vec     = F.(10 .^ range(-2, log10(2.5), length=20))  # log-spaced: 0.01 → 2.5

const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

const N_TRIALS  = 256
const N_SAMP    = 5000
const MIN_PAT   = 500
const MAX_PAT   = 20000

# ──────────────── Heating equilibration ────────────────
const N_EQ_INIT = 100000  # heavy equilibration at T_1 (coldest, near-zero acceptance)
const N_EQ_STEP = 10000   # re-equilibration per subsequent T step

# ──────────────── Adaptive functions ────────────────
function n_patterns(alpha)
    slope = (MAX_PAT - MIN_PAT) / (alpha_vec[end] - alpha_vec[1])
    return MIN_PAT + slope * (alpha - alpha_vec[1])
end

adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── LSR Energy ────────────────
const INF_ENERGY = F(1e30)

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

    # Metropolis accept/reject (unconditionally reject if Ep is infinite — basin escape forbidden)
    CUDA.rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-β * (Ep - E)))
    a3 = reshape(acc, 1, 1, nTrials)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 70)
    println("LSR Alpha Sweep – GPU v3 (Heating protocol, b=$(round(b_lsr, digits=3)))")
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
    total_eq = N_EQ_INIT + (n_T - 1) * N_EQ_STEP
    @printf("Heating protocol: N_EQ_INIT=%d, N_EQ_STEP=%d, total eq=%d\n",
            N_EQ_INIT, N_EQ_STEP, total_eq)
    @printf("Sampling: %d steps per T\n", N_SAMP)
    println("T grid (log-spaced, $n_T points):")
    @printf("  ")
    for j in 1:n_T
        @printf("%.4f ", T_vec[j])
    end
    println("\n")

    # ── Allocate GPU data ──
    println("Allocating GPU memory...")

    pats_g = Vector{CuArray{F,3}}(undef, n_alpha)
    tgts_g = Vector{CuArray{F,3}}(undef, n_alpha)
    ssvec  = Vector{F}(undef, n_alpha)

    # Single state per α (reused across T via heating)
    xs_g   = Vector{CuArray{F,3}}(undef, n_alpha)

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
        xs_g[i]   = CUDA.zeros(F, N, 1, N_TRIALS)
        xp_g[i]   = CUDA.zeros(F, N, 1, N_TRIALS)
        ov_g[i]   = CUDA.zeros(F, P, 1, N_TRIALS)
        E_g[i]    = CUDA.zeros(F, N_TRIALS)
        Ep_g[i]   = CUDA.zeros(F, N_TRIALS)
        ssvec[i]  = adaptive_ss(N)

        mem += (N*P*N_TRIALS + N*N_TRIALS + 2*N*N_TRIALS + P*N_TRIALS + 2*N_TRIALS) * sizeof(F)
    end
    GC.gc()

    ra = CUDA.zeros(F, N_TRIALS)

    # Phi accumulator (reused per T)
    phi_acc = [CUDA.zeros(F, N_TRIALS) for _ in 1:n_alpha]

    @printf("GPU allocated: %.2f GB  (free: %.2f GB)\n\n", mem/1e9, CUDA.available_memory()/1e9)

    # ── Initialize states near targets (for T_1 only) ──
    println("Initializing states near targets...")
    for i in 1:n_alpha
        Nf = F(Ns[i])
        xs_g[i] .= tgts_g[i] .+ F(0.05) .* CUDA.randn(F, Ns[i], 1, N_TRIALS)
        nrm = sqrt.(sum(xs_g[i] .^ 2, dims=1))
        xs_g[i] .= sqrt(Nf) .* xs_g[i] ./ nrm
        compute_energy_lsr_batched!(E_g[i], xs_g[i], pats_g[i], ov_g[i], Nf)
    end
    CUDA.synchronize()
    println("Done.\n")

    # ── Heating protocol: equilibrate + sample at each T, propagate state ──
    println("Heating protocol: T_1=$(T_vec[1]) → T_$(n_T)=$(T_vec[end])")
    println("  Each T: equilibrate → sample → carry state to next T\n")

    phi_grid = zeros(Float64, n_alpha, n_T)
    total_work = total_eq + n_T * N_SAMP
    prog = Progress(total_work, desc="Heating+Sampling: ")
    t0 = time()

    for j in 1:n_T
        T = T_vec[j]
        β = F(1 / T)
        n_eq = (j == 1) ? N_EQ_INIT : N_EQ_STEP

        # ── Equilibration at T_j ──
        # xs_g[i] carries state from T_{j-1} (or initial state if j==1)
        for step in 1:n_eq
            for i in 1:n_alpha
                mc_step_single_T!(xs_g[i], xp_g[i], E_g[i], Ep_g[i],
                                  pats_g[i], ov_g[i], β, ra,
                                  F(Ns[i]), ssvec[i])
            end
            next!(prog)
        end

        # ── Sampling at T_j ──
        for i in 1:n_alpha
            phi_acc[i] .= zero(F)
        end

        for step in 1:N_SAMP
            for i in 1:n_alpha
                Nf = F(Ns[i])
                mc_step_single_T!(xs_g[i], xp_g[i], E_g[i], Ep_g[i],
                                  pats_g[i], ov_g[i], β, ra,
                                  Nf, ssvec[i])
                phi_acc[i] .+= vec(sum(tgts_g[i] .* xs_g[i], dims=1)) ./ Nf
            end
            next!(prog)
        end

        # ── Collect results for T_j ──
        for i in 1:n_alpha
            phi_avg = Array(phi_acc[i]) ./ N_SAMP  # [N_TRIALS]
            phi_grid[i, j] = mean(phi_avg)
        end

        @printf("  T=%.2f (eq=%d): φ(α=%.2f)=%.3f, φ(α=%.2f)=%.3f, φ(α=%.2f)=%.3f\n",
                T, n_eq,
                alpha_vec[1], phi_grid[1, j],
                alpha_vec[n_alpha÷2], phi_grid[n_alpha÷2, j],
                alpha_vec[end], phi_grid[end, j])

        # xs_g[i] carries forward to T_{j+1} — THIS IS THE HEATING PROPAGATION
    end
    finish!(prog)
    CUDA.synchronize()
    t_total = time() - t0
    @printf("\nTotal time: %.1f s (%.2f ms/step avg)\n\n", t_total, 1000*t_total/total_work)

    # ── Save CSV ──
    csv_file = "lsr_heating.csv"
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
    @printf("Total GPU time: %.1f s\n", t_total)
    @printf("Heating protocol: N_EQ_INIT=%d + %d × N_EQ_STEP=%d = %d total eq steps\n",
            N_EQ_INIT, n_T-1, N_EQ_STEP, total_eq)
    @printf("b = %.4f (Epanechnikov kernel)\n", b_lsr)
    println("=" ^ 70)
end

main()
