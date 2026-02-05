#=
Diagnostic script to measure acceptance rates across (α, T) grid
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf

const F = Float32
const b_lsr = F(2 + sqrt(2))
const INF_ENERGY = F(1e30)

# Small grid for quick diagnostics
const alpha_vec = F[0.10, 0.20, 0.25, 0.30, 0.40, 0.50]
const T_vec     = F[0.10, 0.50, 1.00, 1.50, 2.00]
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)
const N_TRIALS  = 64
const N_EQ      = 1000
const N_MEASURE = 500
const MIN_PAT   = 500
const MAX_PAT   = 5000
const n_chains  = n_T * N_TRIALS

function n_patterns(alpha)
    slope = (MAX_PAT - MIN_PAT) / (alpha_vec[end] - alpha_vec[1])
    return MIN_PAT + slope * (alpha - alpha_vec[1])
end

adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

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

# MC step that returns number of acceptances
function mc_step_count!(x::CuArray{F,3}, xp::CuArray{F,3},
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
    n_acc = sum(acc)  # Count acceptances

    a3 = reshape(acc, 1, n_T, N_TRIALS)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)

    return Int(n_acc)
end

function main()
    !CUDA.functional() && error("CUDA not available")
    println("="^60)
    println("LSR Acceptance Rate Diagnostic")
    println("="^60)

    # Compute N, P for each α
    Ns = Int[]; Ps = Int[]
    for α in alpha_vec
        Pt = n_patterns(α)
        N = max(round(Int, log(Pt) / α), 2)
        push!(Ns, N); push!(Ps, round(Int, Pt))
    end

    println("\nGrid setup:")
    for i in 1:n_alpha
        @printf("  α=%.2f: N=%d, P=%d, σ=%.3f\n",
                alpha_vec[i], Ns[i], Ps[i], adaptive_ss(Ns[i]))
    end
    println()

    # Allocate
    pats_g = Vector{CuArray{F,3}}(undef, n_alpha)
    tgts_g = Vector{CuArray{F,3}}(undef, n_alpha)
    xs_g   = Vector{CuArray{F,3}}(undef, n_alpha)
    xps_g  = Vector{CuArray{F,3}}(undef, n_alpha)
    ovs_g  = Vector{CuArray{F,3}}(undef, n_alpha)
    Es_g   = Vector{CuVector{F}}(undef, n_alpha)
    Eps_g  = Vector{CuVector{F}}(undef, n_alpha)
    ssvec  = Vector{F}(undef, n_alpha)

    for i in 1:n_alpha
        N = Ns[i]; P = Ps[i]; Nf = F(N)
        Random.seed!(42 + i)
        p_cpu = randn(F, N, P, N_TRIALS)
        for t in 1:N_TRIALS, j in 1:P
            c = @view p_cpu[:, j, t]
            c .*= sqrt(Nf) / norm(c)
        end
        pats_g[i] = CuArray(p_cpu)
        tgts_g[i] = CuArray(p_cpu[:, 1:1, :])
        xs_g[i]   = CUDA.zeros(F, N, n_T, N_TRIALS)
        xps_g[i]  = CUDA.zeros(F, N, n_T, N_TRIALS)
        ovs_g[i]  = CUDA.zeros(F, P, n_T, N_TRIALS)
        Es_g[i]   = CUDA.zeros(F, n_chains)
        Eps_g[i]  = CUDA.zeros(F, n_chains)
        ssvec[i]  = adaptive_ss(N)
    end

    β_gpu = CuVector{F}(repeat(F.(1 ./ T_vec), N_TRIALS))
    ra = CUDA.zeros(F, n_chains)

    # Initialize near targets
    for i in 1:n_alpha
        Nf = F(Ns[i])
        xs_g[i] .= tgts_g[i] .+ F(0.05) .* CUDA.randn(F, Ns[i], n_T, N_TRIALS)
        nrm = sqrt.(sum(xs_g[i] .^ 2, dims=1))
        xs_g[i] .= sqrt(Nf) .* xs_g[i] ./ nrm
        compute_energy_lsr!(Es_g[i], xs_g[i], pats_g[i], ovs_g[i], Nf)
    end
    CUDA.synchronize()

    # Equilibration
    println("Equilibrating ($N_EQ steps)...")
    for _ in 1:N_EQ
        for i in 1:n_alpha
            mc_step_count!(xs_g[i], xps_g[i], Es_g[i], Eps_g[i],
                           pats_g[i], ovs_g[i], β_gpu, ra,
                           F(Ns[i]), ssvec[i])
        end
    end
    CUDA.synchronize()

    # Measure acceptance rates
    println("Measuring acceptance rates ($N_MEASURE steps)...")
    acc_counts = zeros(Int, n_alpha, n_T)  # [α, T] acceptance counts

    for _ in 1:N_MEASURE
        for i in 1:n_alpha
            n_acc = mc_step_count!(xs_g[i], xps_g[i], Es_g[i], Eps_g[i],
                                   pats_g[i], ovs_g[i], β_gpu, ra,
                                   F(Ns[i]), ssvec[i])
            # n_acc is total across all (T, trial) pairs
            # We want per-T breakdown, so we need to track differently
        end
    end

    # Actually, let's track per-T acceptance by checking energies
    # Reset and measure more carefully
    acc_by_T = zeros(Float64, n_alpha, n_T)

    for step in 1:N_MEASURE
        for i in 1:n_alpha
            E_before = Array(Es_g[i])  # [n_chains]

            mc_step_count!(xs_g[i], xps_g[i], Es_g[i], Eps_g[i],
                           pats_g[i], ovs_g[i], β_gpu, ra,
                           F(Ns[i]), ssvec[i])

            E_after = Array(Es_g[i])
            # Count changes per T
            E_mat_before = reshape(E_before, n_T, N_TRIALS)
            E_mat_after = reshape(E_after, n_T, N_TRIALS)
            for j in 1:n_T
                n_changed = sum(E_mat_before[j, :] .!= E_mat_after[j, :])
                acc_by_T[i, j] += n_changed
            end
        end
    end
    CUDA.synchronize()

    acc_by_T ./= (N_MEASURE * N_TRIALS)  # Convert to rates

    # Print results
    println("\n" * "="^60)
    println("ACCEPTANCE RATES (fraction accepted)")
    println("="^60)

    # Header
    print("α\\T   ")
    for T in T_vec
        @printf("%6.2f ", T)
    end
    println("\n" * "-"^60)

    for i in 1:n_alpha
        @printf("%.2f  ", alpha_vec[i])
        for j in 1:n_T
            rate = acc_by_T[i, j]
            # Color coding: green=good, yellow=marginal, red=bad
            if 0.2 <= rate <= 0.5
                @printf("\e[32m%5.1f%%\e[0m ", 100*rate)  # green
            elseif 0.1 <= rate < 0.2 || 0.5 < rate <= 0.7
                @printf("\e[33m%5.1f%%\e[0m ", 100*rate)  # yellow
            else
                @printf("\e[31m%5.1f%%\e[0m ", 100*rate)  # red
            end
        end
        @printf(" N=%d σ=%.2f\n", Ns[i], ssvec[i])
    end

    println("\n" * "-"^60)
    println("Legend: \e[32mgreen\e[0m=optimal(20-50%), \e[33myellow\e[0m=marginal, \e[31mred\e[0m=poor")
    println("="^60)

    # Also measure φ to see current state
    println("\nCurrent φ values (retrieval order parameter):")
    print("α\\T   ")
    for T in T_vec
        @printf("%6.2f ", T)
    end
    println()

    for i in 1:n_alpha
        Nf = F(Ns[i])
        phi = Array(vec(sum(tgts_g[i] .* xs_g[i], dims=1))) ./ Nf
        phi_mat = reshape(phi, n_T, N_TRIALS)
        phi_avg = vec(mean(phi_mat, dims=2))

        @printf("%.2f  ", alpha_vec[i])
        for j in 1:n_T
            @printf("%5.2f  ", phi_avg[j])
        end
        println()
    end
end

main()
