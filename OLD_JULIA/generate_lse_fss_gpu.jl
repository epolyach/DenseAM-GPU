#=
GPU-Accelerated LSE Alpha Sweep – Finite-Size Scaling
Runs additional P scales beyond the base (lse_alpha_sweep_table.csv).
Sequential α processing to handle large P values.
Output: lse_s2.csv, lse_s3.csv
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter

const F = Float32
const betanet = F(1.0)

const alpha_vec = collect(F(0.01):F(0.01):F(0.55))
const T_vec     = collect(F(0.05):F(0.05):F(2.50))
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)
const N_EQ      = 5000
const N_SAMP    = 5000

adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── LSE Energy ────────────────
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

# ──────────────── MC Step ────────────────
function mc_step!(x::CuArray{F,3}, xp::CuArray{F,3},
                  E::CuVector{F}, Ep::CuVector{F},
                  pat::CuArray{F,3}, ov::CuArray{F,3},
                  β::CuVector{F}, ra::CuVector{F},
                  Nf::F, ss::F, nT::Int, nTr::Int)
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lse!(Ep, xp, pat, ov, Nf)

    CUDA.rand!(ra)
    acc = @. ra < exp(-(β * (Ep - E)))
    a3 = reshape(acc, 1, nT, nTr)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# ──────────────── Run one scale ────────────────
function run_scale(min_pat::Int, max_pat::Int, n_trials::Int, csv_file::String)
    n_chains = n_T * n_trials

    println("─" ^ 60)
    @printf("Scale: P = %d – %d,  trials = %d\n", min_pat, max_pat, n_trials)
    println("─" ^ 60)

    # Compute N(α), P(α)
    Ns = Int[]; Ps = Int[]
    for α in alpha_vec
        slope = (max_pat - min_pat) / (alpha_vec[end] - alpha_vec[1])
        Pt = min_pat + slope * (α - alpha_vec[1])
        N  = max(round(Int, log(Pt) / α), 2)
        push!(Ns, N); push!(Ps, round(Int, Pt))
    end
    @printf("  N range: %d – %d,  P range: %d – %d\n", extrema(Ns)..., extrema(Ps)...)

    # Shared buffers
    β_gpu = CuVector{F}(repeat(F.(1 ./ T_vec), n_trials))
    ra    = CUDA.zeros(F, n_chains)

    phi_grid = zeros(Float64, n_alpha, n_T)

    prog = Progress(n_alpha, desc="  α sweep: ")
    t0 = time()

    for i in 1:n_alpha
        N = Ns[i]; P = Ps[i]; Nf = F(N)
        ss = adaptive_ss(N)

        # Generate normalised patterns on CPU
        Random.seed!(42 + i)
        p_cpu = randn(F, N, P, n_trials)
        for t in 1:n_trials, j in 1:P
            c = @view p_cpu[:, j, t]
            c .*= sqrt(Nf) / norm(c)
        end

        # Transfer to GPU
        pat = CuArray(p_cpu)
        tgt = CuArray(p_cpu[:, 1:1, :])
        x   = CUDA.zeros(F, N, n_T, n_trials)
        xp  = CUDA.zeros(F, N, n_T, n_trials)
        ov  = CUDA.zeros(F, P, n_T, n_trials)
        E   = CUDA.zeros(F, n_chains)
        Ep  = CUDA.zeros(F, n_chains)
        phi_acc = CUDA.zeros(F, n_chains)

        p_cpu = nothing; GC.gc()  # free CPU patterns

        # Initialize near target
        x .= tgt .+ F(0.05) .* CUDA.randn(F, N, n_T, n_trials)
        nrm = sqrt.(sum(x .^ 2, dims=1))
        x .= sqrt(Nf) .* x ./ nrm
        compute_energy_lse!(E, x, pat, ov, Nf)

        # Equilibration
        for step in 1:N_EQ
            mc_step!(x, xp, E, Ep, pat, ov, β_gpu, ra, Nf, ss, n_T, n_trials)
        end

        # Sampling
        phi_acc .= zero(F)
        for step in 1:N_SAMP
            mc_step!(x, xp, E, Ep, pat, ov, β_gpu, ra, Nf, ss, n_T, n_trials)
            phi_acc .+= vec(sum(tgt .* x, dims=1)) ./ Nf
        end

        CUDA.synchronize()

        # Collect
        phi_avg = Array(phi_acc) ./ N_SAMP
        phi_mat = reshape(phi_avg, n_T, n_trials)
        phi_grid[i, :] = vec(mean(phi_mat, dims=2))

        # Free GPU memory
        CUDA.unsafe_free!(pat)
        CUDA.unsafe_free!(tgt)
        CUDA.unsafe_free!(x)
        CUDA.unsafe_free!(xp)
        CUDA.unsafe_free!(ov)
        CUDA.unsafe_free!(E)
        CUDA.unsafe_free!(Ep)
        CUDA.unsafe_free!(phi_acc)
        GC.gc(); CUDA.reclaim()

        next!(prog)
    end
    finish!(prog)
    elapsed = time() - t0
    @printf("  Scale done: %.1f s (%.1f s/α)\n", elapsed, elapsed/n_alpha)

    # Free shared buffers
    CUDA.unsafe_free!(β_gpu)
    CUDA.unsafe_free!(ra)

    # Save CSV
    open(csv_file, "w") do f
        write(f, "alpha")
        for T in T_vec; write(f, @sprintf(",T%.2f", T)); end
        write(f, "\n")
        for i in 1:n_alpha
            write(f, @sprintf("%.2f", alpha_vec[i]))
            for j in 1:n_T; write(f, @sprintf(",%.4f", phi_grid[i, j])); end
            write(f, "\n")
        end
    end
    println("  CSV saved: $csv_file")

    # Sample output
    for idx in [1, n_alpha÷2, n_alpha]
        j_1 = findfirst(t -> t ≈ F(1.0), T_vec)
        @printf("    α=%.2f (N=%d, P=%d): φ(T=0.05)=%.4f, φ(T=1.00)=%.4f, φ(T=2.50)=%.4f\n",
                alpha_vec[idx], Ns[idx], Ps[idx],
                phi_grid[idx, 1], phi_grid[idx, j_1], phi_grid[idx, end])
    end
    println()
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 70)
    println("LSE Finite-Size Scaling – GPU")
    println("=" ^ 70)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # Scale 2: moderate P
    run_scale(2000, 50000, 256, "lse_s2.csv")

    # Scale 3: large P, fewer trials
    run_scale(10000, 300000, 64, "lse_s3.csv")

    println("=" ^ 70)
    println("All LSE scales complete.")
    println("=" ^ 70)
end

main()
