#=
GPU-Accelerated LSR Alpha Sweep – Correct Support Boundary
- Epanechnikov kernel with b = 2 + √2, support φ ∈ [1-1/b, 1]
- IMPORTANT: Steps outside all pattern supports have INFINITE energy
  and are always rejected. No artificial clamping or tunneling.
- The support boundary is an impenetrable barrier — once inside a
  pattern's basin, the MC cannot escape via local moves.
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

# const alpha_vec = collect(F(0.01):F(0.01):F(0.55))  # 55 values
# const T_vec     = collect(F(0.05):F(0.05):F(2.50))   # 50 values

const alpha_vec = collect(F(0.2):F(0.02):F(0.54))  
const T_vec     = collect(F(0.05):F(0.05):F(1.50))  

const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)
const N_TRIALS  = 64 # 256
const N_SAMP    = 500 # 5000
const MIN_PAT   = 500 # 200
const MAX_PAT   = 20000 # 5000
const n_chains  = n_T * N_TRIALS  # 12800

# ──────────────── T-dependent equilibration ────────────────
# Keep constant number of accepted moves across T
# N_EQ(T) = N_EQ_base * exp(c * (1/T - 1/T_max)), capped at N_EQ_cap
const N_EQ_BASE = 5000
const N_EQ_CAP  = 300000
const EQ_COEFF  = F(0.15)  # empirical coefficient from acceptance measurements
const T_MAX     = maximum(T_vec)

function n_eq_for_T(T::Real)
    n_eq = N_EQ_BASE * exp(EQ_COEFF * (1/T - 1/T_MAX))
    return min(N_EQ_CAP, round(Int, n_eq))
end

# Compute N_EQ for each T and find maximum (needed for loop)
const N_EQ_vec = [n_eq_for_T(T) for T in T_vec]
const N_EQ_MAX = maximum(N_EQ_vec)

# Adaptive pattern count (linear in α)
function n_patterns(alpha)
    slope = (MAX_PAT - MIN_PAT) / (alpha_vec[end] - alpha_vec[1])
    return MIN_PAT + slope * (alpha - alpha_vec[1])
end

# Adaptive step size
adaptive_ss(N::Int) = max(F(0.1), F(2.4) / sqrt(F(N)))

# ──────────────── LSR Energy (batched over trials) ────────────────
# IMPORTANT: If x is outside the support of ALL patterns, energy = +∞
# This makes such moves always rejected by Metropolis.
const INF_ENERGY = F(1e30)  # Effectively infinite energy

function compute_energy_lsr!(E::CuVector{F}, x::CuArray{F,3},
                              patterns::CuArray{F,3}, overlap::CuArray{F,3},
                              Nf::F)
    Nb = Nf / b_lsr

    # Batched gemm: overlap[p,j,t] = Σ_n patterns[n,p,t] * x[n,j,t]
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)

    # LSR: E = -(N/b) * log(Σ_p max(0, 1 - b + b*overlap/N))
    # Support boundary: φ = overlap/N ≥ 1-1/b, i.e., overlap ≥ N*(1-1/b)
    @. overlap = max(zero(F), one(F) - b_lsr + b_lsr * overlap / Nf)
    s = sum(overlap, dims=1)                        # [1 × n_T × N_TRIALS]

    # If s = 0, we are outside all supports → infinite energy (move rejected)
    # If s > 0, compute normal energy
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

    # Metropolis accept/reject
    CUDA.rand!(ra)
    acc = @. ra < exp(-(β * (Ep - E)))
    a3 = reshape(acc, 1, n_T, N_TRIALS)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# MC Step with active mask — only update chains where mask is true
function mc_step_masked!(x::CuArray{F,3}, xp::CuArray{F,3},
                         E::CuVector{F}, Ep::CuVector{F},
                         pat::CuArray{F,3}, ov::CuArray{F,3},
                         β::CuVector{F}, ra::CuVector{F},
                         active::CuArray{Bool,3},  # [1, n_T, N_TRIALS]
                         Nf::F, ss::F)
    nT = size(x, 2)
    nTrials = size(x, 3)

    # Propose on sphere
    CUDA.randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    # Proposed energy
    compute_energy_lsr!(Ep, xp, pat, ov, Nf)

    # Metropolis accept/reject, but only for active chains
    CUDA.rand!(ra)
    acc = @. ra < exp(-(β * (Ep - E)))
    a3 = reshape(acc, 1, nT, nTrials)

    # Only update if active
    @. x = ifelse(active & a3, xp, x)

    # For E: reshape to 3D, apply mask, then flatten back
    E3 = reshape(E, 1, nT, nTrials)
    Ep3 = reshape(Ep, 1, nT, nTrials)
    @. E3 = ifelse(active & a3, Ep3, E3)
    return nothing
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 70)
    println("LSR Alpha Sweep – GPU (T-dependent Equilibration, b=$(round(b_lsr, digits=3)))")
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
    @printf("Grid: %d α × %d T,  trials: %d,  MC: %d samp\n", n_alpha, n_T, N_TRIALS, N_SAMP)
    @printf("T-dependent equilibration: N_EQ(T) = %d × exp(%.2f × (1/T - 1/%.1f)), cap=%d\n",
            N_EQ_BASE, EQ_COEFF, T_MAX, N_EQ_CAP)
    @printf("  T=%.2f: N_EQ=%d,  T=%.2f: N_EQ=%d,  T=%.2f: N_EQ=%d\n",
            T_vec[end], N_EQ_vec[end], T_vec[n_T÷2], N_EQ_vec[n_T÷2], T_vec[1], N_EQ_vec[1])
    @printf("  → Running %d total eq steps (determined by T_min=%.2f)\n\n", N_EQ_MAX, T_vec[1])

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

    # ── Equilibration (T-dependent: each T gets N_EQ(T) steps) ──
    println("Equilibration (T-dependent, max $N_EQ_MAX steps)...")
    @printf("  Each T gets its own N_EQ: T=%.2f→%d, T=%.2f→%d, T=%.2f→%d\n",
            T_vec[1], N_EQ_vec[1], T_vec[n_T÷2], N_EQ_vec[n_T÷2], T_vec[end], N_EQ_vec[end])

    # Create active mask on CPU, transfer once
    active_cpu = ones(Bool, 1, n_T, N_TRIALS)
    active_g = Vector{CuArray{Bool,3}}(undef, n_alpha)
    for i in 1:n_alpha
        active_g[i] = CuArray(active_cpu)
    end

    # Find unique N_EQ thresholds (when T values should stop)
    thresholds = sort(unique(N_EQ_vec))

    t0 = time()
    current_threshold_idx = 1
    step = 0
    total_steps = N_EQ_MAX

    prog = Progress(total_steps, desc="Equilibration: ")
    while step < N_EQ_MAX
        # Determine how many steps until next threshold
        next_threshold = current_threshold_idx <= length(thresholds) ? thresholds[current_threshold_idx] : N_EQ_MAX
        steps_this_phase = min(next_threshold, N_EQ_MAX) - step

        # Run this phase
        for _ in 1:steps_this_phase
            step += 1
            for i in 1:n_alpha
                mc_step_masked!(xs_g[i], xps_g[i], Es_g[i], Eps_g[i],
                                pats_g[i], ovs_g[i], β_gpu, ra,
                                active_g[i], F(Ns[i]), ssvec[i])
            end
            next!(prog)
        end

        # Update mask: deactivate T values that reached their quota
        if current_threshold_idx <= length(thresholds)
            threshold = thresholds[current_threshold_idx]
            n_deactivated = 0
            for j in 1:n_T
                if N_EQ_vec[j] <= threshold && active_cpu[1, j, 1]
                    active_cpu[1, j, :] .= false
                    n_deactivated += 1
                end
            end
            # Transfer updated mask to GPU
            for i in 1:n_alpha
                copyto!(active_g[i], active_cpu)
            end
            n_still_active = sum(active_cpu[1, :, 1])
            @printf("\r  Step %d: deactivated %d T values, %d still active    \n",
                    step, n_deactivated, n_still_active)
            current_threshold_idx += 1
        end
    end
    finish!(prog)
    CUDA.synchronize()
    t_eq = time() - t0
    @printf("Equilibration: %.1f s (%.2f ms/step)\n\n", t_eq, 1000*t_eq/N_EQ_MAX)

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

    # ── Save CSV (identical format to CPU version) ──
    csv_file = "lsr_longeq.csv"
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

    # Sample output
    println("Sample data:")
    for idx in [1, n_alpha÷2, n_alpha]
        j_1 = findfirst(t -> t ≈ F(1.0), T_vec)
        @printf("  α=%.2f (N=%d, P=%d): φ(T=0.05)=%.4f, φ(T=1.00)=%.4f, φ(T=2.50)=%.4f\n",
                alpha_vec[idx], Ns[idx], Ps[idx],
                phi_grid[idx, 1], phi_grid[idx, j_1], phi_grid[idx, end])
    end
    println()
    @printf("Total GPU time: %.1f s (eq: %.1f + samp: %.1f)\n", t_eq+t_samp, t_eq, t_samp)
    @printf("Equilibration: %d steps (T-dependent, base=%d, cap=%d)\n", N_EQ_MAX, N_EQ_BASE, N_EQ_CAP)
    @printf("b = %.4f (Epanechnikov kernel)\n", b_lsr)
    println("=" ^ 70)
end

main()
