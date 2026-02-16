#=
Acceptance rate map in (α, T) plane
Test different σ(T) dependencies
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using Plots

const F = Float32
const b_lsr = F(2 + sqrt(2))
const INF_ENERGY = F(1e30)

# Grid (coarser for speed)
const alpha_vec = collect(F(0.2):F(0.05):F(0.5))  # 11 values
const T_vec     = collect(F(0.02):F(0.02):F(0.50))  # 25 values
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)
const N_TRIALS  = 32
const N_EQ      = 500
const N_MEASURE = 200
const MIN_PAT   = 500
const MAX_PAT   = 10000
const n_chains  = n_T * N_TRIALS

function n_patterns(alpha)
    slope = (MAX_PAT - MIN_PAT) / (alpha_vec[end] - alpha_vec[1])
    return MIN_PAT + slope * (alpha - alpha_vec[1])
end

# ──────────────── Step size functions ────────────────
# Option 1: No T dependence (current formula)
σ_const(N, T) = max(F(0.1), F(2.4) / sqrt(F(N)))

# Option 2: σ ∝ √T — for low T region
# At low T, acceptance ~ exp(-ΔE/T), need smaller steps
σ_sqrt_T(N, T) = max(F(0.02), F(2.0) * sqrt(T) / sqrt(F(N)))

# Option 3: σ ∝ T — even more aggressive scaling
σ_linear_T(N, T) = max(F(0.01), F(1.5) * T / sqrt(F(N)))

# Choose which one to use
const σ_func = σ_sqrt_T  # √T scaling for low T region

# ──────────────── Energy ────────────────
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

# ──────────────── MC Step with per-T step sizes ────────────────
function mc_step_with_ss!(x::CuArray{F,3}, xp::CuArray{F,3},
                          E::CuVector{F}, Ep::CuVector{F},
                          pat::CuArray{F,3}, ov::CuArray{F,3},
                          β::CuVector{F}, ra::CuVector{F},
                          ss_vec::CuVector{F},  # [n_T] step sizes
                          Nf::F)
    # Generate noise
    CUDA.randn!(xp)

    # Apply per-T step sizes: ss_vec is [n_T], broadcast over [N, n_T, N_TRIALS]
    ss_3d = reshape(ss_vec, 1, n_T, 1)
    @. xp = x + ss_3d * xp

    # Normalize to sphere
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lsr!(Ep, xp, pat, ov, Nf)

    CUDA.rand!(ra)
    acc = @. (Ep < INF_ENERGY) & (ra < exp(-(β * (Ep - E))))
    a3 = reshape(acc, 1, n_T, N_TRIALS)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)

    return acc  # Return acceptance mask
end

function main()
    !CUDA.functional() && error("CUDA not available")
    println("="^60)
    println("Acceptance Rate Map — σ = $(σ_func)")
    println("="^60)

    # Compute N, P for each α
    Ns = Int[]; Ps = Int[]
    for α in alpha_vec
        Pt = n_patterns(α)
        N = max(round(Int, log(Pt) / α), 2)
        push!(Ns, N); push!(Ps, round(Int, Pt))
    end

    # Build step size vectors for each α (varies with T)
    ss_vecs = Vector{CuVector{F}}(undef, n_alpha)
    for i in 1:n_alpha
        N = Ns[i]
        ss_cpu = F[σ_func(N, T) for T in T_vec]
        ss_vecs[i] = CuVector{F}(ss_cpu)
    end

    # Print step sizes
    println("\nStep sizes σ(N, T):")
    println("α     N    T=0.1  T=1.0  T=2.5")
    for i in 1:n_alpha
        ss_cpu = Array(ss_vecs[i])
        @printf("%.2f  %3d  %.3f  %.3f  %.3f\n",
                alpha_vec[i], Ns[i], ss_cpu[1], ss_cpu[10], ss_cpu[end])
    end
    println()

    # Allocate GPU arrays
    pats_g = Vector{CuArray{F,3}}(undef, n_alpha)
    tgts_g = Vector{CuArray{F,3}}(undef, n_alpha)
    xs_g   = Vector{CuArray{F,3}}(undef, n_alpha)
    xps_g  = Vector{CuArray{F,3}}(undef, n_alpha)
    ovs_g  = Vector{CuArray{F,3}}(undef, n_alpha)
    Es_g   = Vector{CuVector{F}}(undef, n_alpha)
    Eps_g  = Vector{CuVector{F}}(undef, n_alpha)

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
    end

    β_gpu = CuVector{F}(repeat(F.(1 ./ T_vec), N_TRIALS))
    ra = CUDA.zeros(F, n_chains)

    # Initialize near targets
    println("Initializing...")
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
            mc_step_with_ss!(xs_g[i], xps_g[i], Es_g[i], Eps_g[i],
                             pats_g[i], ovs_g[i], β_gpu, ra,
                             ss_vecs[i], F(Ns[i]))
        end
    end
    CUDA.synchronize()

    # Measure acceptance rates
    println("Measuring acceptance ($N_MEASURE steps)...")
    acc_counts = zeros(Float64, n_alpha, n_T)

    for _ in 1:N_MEASURE
        for i in 1:n_alpha
            acc_mask = mc_step_with_ss!(xs_g[i], xps_g[i], Es_g[i], Eps_g[i],
                                        pats_g[i], ovs_g[i], β_gpu, ra,
                                        ss_vecs[i], F(Ns[i]))
            # acc_mask is [n_chains] = [n_T * N_TRIALS]
            acc_cpu = Array(acc_mask)
            acc_mat = reshape(acc_cpu, n_T, N_TRIALS)
            for j in 1:n_T
                acc_counts[i, j] += sum(acc_mat[j, :])
            end
        end
    end
    CUDA.synchronize()

    acc_rates = acc_counts ./ (N_MEASURE * N_TRIALS)

    # Print table
    println("\n" * "="^60)
    println("ACCEPTANCE RATES")
    println("="^60)

    # Also measure φ
    phi_grid = zeros(Float64, n_alpha, n_T)
    for i in 1:n_alpha
        Nf = F(Ns[i])
        phi = Array(vec(sum(tgts_g[i] .* xs_g[i], dims=1))) ./ Nf
        phi_mat = reshape(phi, n_T, N_TRIALS)
        phi_grid[i, :] = vec(mean(phi_mat, dims=2))
    end

    # Create heatmap - use actual data range for acceptance
    acc_min, acc_max = minimum(acc_rates), maximum(acc_rates)
    cmap_acc = cgrad([:darkred, :red, :orange, :yellow, :green])

    p1 = heatmap(alpha_vec, T_vec, acc_rates',
                 xlabel="α", ylabel="T",
                 title="Acceptance Rate ($(round(100*acc_min,digits=1))% - $(round(100*acc_max,digits=1))%)",
                 color=cmap_acc, clims=(acc_min, acc_max),
                 colorbar_title="%",
                 size=(600, 500))

    # Add contours at meaningful levels within the data range
    mid_level = (acc_min + acc_max) / 2
    contour!(p1, alpha_vec, T_vec, acc_rates',
             levels=[mid_level],
             color=:black, linewidth=1.5, label=false)

    # φ panel: shows whether system is in retrieval (φ≈1) or disordered (φ≈0)
    p2 = heatmap(alpha_vec, T_vec, phi_grid',
                 xlabel="α", ylabel="T",
                 title="φ = ⟨target·x⟩/N (retrieval if ≈1)",
                 color=cgrad([:darkblue, :blue, :white, :red, :darkred], rev=true),
                 clims=(0, 1),
                 size=(600, 500))

    fig = plot(p1, p2, layout=(1, 2), size=(1200, 500))
    savefig(fig, "acceptance_map.png")
    println("\nSaved: acceptance_map.png")

    # Print summary statistics
    println("\nAcceptance rate statistics:")
    @printf("  Min: %.1f%%  Max: %.1f%%  Mean: %.1f%%\n",
            100*minimum(acc_rates), 100*maximum(acc_rates), 100*mean(acc_rates))

    # Find problematic regions
    low_acc = findall(acc_rates .< 0.15)
    high_acc = findall(acc_rates .> 0.6)

    if !isempty(low_acc)
        println("\nLow acceptance (<15%):")
        for idx in low_acc[1:min(5, length(low_acc))]
            @printf("  α=%.2f, T=%.1f: %.1f%%\n",
                    alpha_vec[idx[1]], T_vec[idx[2]], 100*acc_rates[idx])
        end
    end

    if !isempty(high_acc)
        println("\nHigh acceptance (>60%):")
        for idx in high_acc[1:min(5, length(high_acc))]
            @printf("  α=%.2f, T=%.1f: %.1f%%\n",
                    alpha_vec[idx[1]], T_vec[idx[2]], 100*acc_rates[idx])
        end
    end

    println("\n" * "="^60)
end

main()
