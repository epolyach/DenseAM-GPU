#=
GPU-Accelerated LSE Alpha Sweep – CUDA Streams + Fine Grid (v4)
- CUDA streams for concurrent α processing → better GPU utilization
- Double-buffered random numbers: overlap RNG with compute
- Finer α grid (0.01:0.01:0.55, 55 values) and T grid (50 log-spaced)
- Heating protocol: start from coldest T, carry state forward
- Output: lse_heating.csv
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

const F = Float32

# ──────────────── Parameters ────────────────
const betanet   = F(1.0)

const alpha_vec = collect(F(0.01):F(0.01):F(0.55))
const T_vec     = F.(10 .^ range(-2, log10(2.5), length=50))  # log-spaced: 0.01 → 2.5

const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

const N_TRIALS  = 64
const N_SAMP    = 500
const MIN_PAT   = 500
const MAX_PAT   = 20000

# ──────────────── Heating equilibration ────────────────
const N_EQ_INIT = 300000  # heavy equilibration at T_1 (coldest)
const N_EQ_STEP = 30000   # re-equilibration per subsequent T step

# ──────────────── Adaptive functions ────────────────
function n_patterns(alpha)
    slope = (MAX_PAT - MIN_PAT) / (alpha_vec[end] - alpha_vec[1])
    return MIN_PAT + slope * (alpha - alpha_vec[1])
end

adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── LSE Energy ────────────────
function compute_energy_lse_batched!(E::CuVector{F}, x::CuArray{F,3},
                                      patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                      Nf::F)
    # overlap[p,1,t] = Σ_n patterns[n,p,t] * x[n,1,t]
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)

    # Log-sum-exp trick
    @. overlap = -betanet * (Nf - overlap)
    m = maximum(overlap, dims=1)
    @. overlap = exp(overlap - m)
    s = sum(overlap, dims=1)
    E .= vec(@. -(m + log(s)) / betanet)
    return nothing
end

# ──────────────── MC Step with pre-generated randoms ────────────────
function mc_step_prerand!(x::CuArray{F,3}, xp::CuArray{F,3},
                          E::CuVector{F}, Ep::CuVector{F},
                          pat::CuArray{F,3}, ov::CuArray{F,3},
                          β::F, ra::CuVector{F},
                          Nf::F, ss::F)
    nTrials = size(x, 3)

    # Construct proposal on sphere
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    # Proposed energy
    compute_energy_lse_batched!(Ep, xp, pat, ov, Nf)

    # Metropolis accept/reject
    acc = @. ra < exp(-β * (Ep - E))
    a3 = reshape(acc, 1, 1, nTrials)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Streamed MC step helpers ────────────────
function fill_randoms!(xp_buf, ra_buf, n_alpha)
    for i in 1:n_alpha
        CUDA.randn!(xp_buf[i])
        CUDA.rand!(ra_buf[i])
    end
end

function dispatch_eq_step!(streams, xs_g, xp_buf, E_g, Ep_g, pats_g, ov_g,
                           ra_buf, β, Ns, ssvec, n_alpha)
    for i in 1:n_alpha
        CUDA.stream!(streams[i]) do
            mc_step_prerand!(xs_g[i], xp_buf[i], E_g[i], Ep_g[i],
                            pats_g[i], ov_g[i], β, ra_buf[i],
                            F(Ns[i]), ssvec[i])
        end
    end
end

function dispatch_samp_step!(streams, xs_g, xp_buf, E_g, Ep_g, pats_g, ov_g,
                             ra_buf, β, Ns, ssvec, tgts_g, phi_acc, n_alpha)
    for i in 1:n_alpha
        Nf = F(Ns[i])
        CUDA.stream!(streams[i]) do
            mc_step_prerand!(xs_g[i], xp_buf[i], E_g[i], Ep_g[i],
                            pats_g[i], ov_g[i], β, ra_buf[i],
                            Nf, ssvec[i])
            phi_acc[i] .+= vec(sum(tgts_g[i] .* xs_g[i], dims=1)) ./ Nf
        end
    end
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 70)
    println("LSE Alpha Sweep – GPU v4 (Streams + Fine Grid, βnet=$(betanet))")
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
    @printf("CUDA streams: %d (one per α, double-buffered RNG)\n", n_alpha)
    println("T grid (log-spaced, $n_T points):")
    @printf("  ")
    for j in 1:n_T
        @printf("%.4f ", T_vec[j])
    end
    println("\n")

    # ── Create CUDA streams ──
    streams = [CuStream() for _ in 1:n_alpha]

    # ── Allocate GPU data ──
    println("Allocating GPU memory...")

    pats_g = Vector{CuArray{F,3}}(undef, n_alpha)
    tgts_g = Vector{CuArray{F,3}}(undef, n_alpha)
    ssvec  = Vector{F}(undef, n_alpha)

    xs_g   = Vector{CuArray{F,3}}(undef, n_alpha)
    ov_g   = Vector{CuArray{F,3}}(undef, n_alpha)
    E_g    = Vector{CuVector{F}}(undef, n_alpha)
    Ep_g   = Vector{CuVector{F}}(undef, n_alpha)

    # Double-buffered arrays for streaming (A/B alternation)
    xp_A   = Vector{CuArray{F,3}}(undef, n_alpha)
    xp_B   = Vector{CuArray{F,3}}(undef, n_alpha)
    ra_A   = Vector{CuVector{F}}(undef, n_alpha)
    ra_B   = Vector{CuVector{F}}(undef, n_alpha)

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
        xp_A[i]   = CUDA.zeros(F, N, 1, N_TRIALS)
        xp_B[i]   = CUDA.zeros(F, N, 1, N_TRIALS)
        ov_g[i]   = CUDA.zeros(F, P, 1, N_TRIALS)
        E_g[i]    = CUDA.zeros(F, N_TRIALS)
        Ep_g[i]   = CUDA.zeros(F, N_TRIALS)
        ra_A[i]   = CUDA.zeros(F, N_TRIALS)
        ra_B[i]   = CUDA.zeros(F, N_TRIALS)
        ssvec[i]  = adaptive_ss(N)

        mem += (N*P*N_TRIALS + N*N_TRIALS + N*N_TRIALS + 2*N*N_TRIALS + P*N_TRIALS + 2*N_TRIALS + 2*N_TRIALS) * sizeof(F)
    end
    GC.gc()

    phi_acc = [CUDA.zeros(F, N_TRIALS) for _ in 1:n_alpha]

    @printf("GPU allocated: %.2f GB  (free: %.2f GB)\n\n", mem/1e9, CUDA.available_memory()/1e9)

    # ── Initialize states near targets (for T_1 only) ──
    println("Initializing states near targets...")
    for i in 1:n_alpha
        Nf = F(Ns[i])
        xs_g[i] .= tgts_g[i] .+ F(0.05) .* CUDA.randn(F, Ns[i], 1, N_TRIALS)
        nrm = sqrt.(sum(xs_g[i] .^ 2, dims=1))
        xs_g[i] .= sqrt(Nf) .* xs_g[i] ./ nrm
        compute_energy_lse_batched!(E_g[i], xs_g[i], pats_g[i], ov_g[i], Nf)
    end
    CUDA.synchronize()
    println("Done.\n")

    # ── Heating protocol with CUDA streams ──
    println("Heating protocol: T_1=$(T_vec[1]) → T_$(n_T)=$(T_vec[end])")
    println("  Each T: equilibrate → sample → carry state to next T")
    println("  $(n_alpha) CUDA streams, double-buffered RNG\n")

    phi_grid = zeros(Float64, n_alpha, n_T)
    csv_file = "lse_heating.csv"
    total_work = total_eq + n_T * N_SAMP
    prog = Progress(total_work, desc="Heating+Sampling: ")
    t0 = time()

    for j in 1:n_T
        T = T_vec[j]
        β = F(1 / T)
        n_eq = (j == 1) ? N_EQ_INIT : N_EQ_STEP

        # ── Equilibration at T_j (streamed, double-buffered) ──
        xp_cur, xp_nxt = xp_A, xp_B
        ra_cur, ra_nxt = ra_A, ra_B

        # Pre-fill first buffer
        fill_randoms!(xp_cur, ra_cur, n_alpha)
        CUDA.synchronize()

        for step in 1:n_eq
            dispatch_eq_step!(streams, xs_g, xp_cur, E_g, Ep_g, pats_g, ov_g,
                             ra_cur, β, Ns, ssvec, n_alpha)

            fill_randoms!(xp_nxt, ra_nxt, n_alpha)

            CUDA.synchronize()

            xp_cur, xp_nxt = xp_nxt, xp_cur
            ra_cur, ra_nxt = ra_nxt, ra_cur
            next!(prog)
        end

        # ── Sampling at T_j (streamed, double-buffered) ──
        for i in 1:n_alpha
            phi_acc[i] .= zero(F)
        end
        CUDA.synchronize()

        fill_randoms!(xp_cur, ra_cur, n_alpha)
        CUDA.synchronize()

        for step in 1:N_SAMP
            dispatch_samp_step!(streams, xs_g, xp_cur, E_g, Ep_g, pats_g, ov_g,
                               ra_cur, β, Ns, ssvec, tgts_g, phi_acc, n_alpha)

            fill_randoms!(xp_nxt, ra_nxt, n_alpha)

            CUDA.synchronize()
            xp_cur, xp_nxt = xp_nxt, xp_cur
            ra_cur, ra_nxt = ra_nxt, ra_cur
            next!(prog)
        end

        # ── Collect results for T_j ──
        for i in 1:n_alpha
            phi_avg = Array(phi_acc[i]) ./ N_SAMP
            phi_grid[i, j] = mean(phi_avg)
        end

        @printf("  T=%.4f (eq=%d): φ(α=%.2f)=%.3f, φ(α=%.2f)=%.3f, φ(α=%.2f)=%.3f\n",
                T, n_eq,
                alpha_vec[1], phi_grid[1, j],
                alpha_vec[n_alpha÷2], phi_grid[n_alpha÷2, j],
                alpha_vec[end], phi_grid[end, j])

        # ── Write CSV with columns completed so far ──
        open(csv_file, "w") do f
            write(f, "alpha")
            for jj in 1:j
                write(f, @sprintf(",T%.4f", T_vec[jj]))
            end
            write(f, "\n")
            for i in 1:n_alpha
                write(f, @sprintf("%.2f", alpha_vec[i]))
                for jj in 1:j
                    write(f, @sprintf(",%.4f", phi_grid[i, jj]))
                end
                write(f, "\n")
            end
        end

        # xs_g[i] carries forward to T_{j+1} — HEATING PROPAGATION
    end
    finish!(prog)
    CUDA.synchronize()
    t_total = time() - t0
    @printf("\nTotal time: %.1f s (%.2f ms/step avg)\n\n", t_total, 1000*t_total/total_work)

    println("CSV saved: $csv_file ($n_T T columns)")
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
    @printf("CUDA streams: %d concurrent α chains (double-buffered)\n", n_alpha)
    @printf("βnet = %.1f (log-sum-exp kernel)\n", betanet)
    println("=" ^ 70)
end

main()
