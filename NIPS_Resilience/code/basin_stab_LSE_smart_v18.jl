#=
GPU-Accelerated Smart-MC LSE Basin Stability Test (v18) — PER-SAMPLE OUTPUT
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSE_smart_v18.jl           # resume from last completed α
  julia basin_stab_LSE_smart_v18.jl --fresh   # overwrite CSV, start from scratch

Goal (AAAI paper): Numerically confirm the exact-density LSE retrieval
  boundary α_c^E(T) = -(1/2) ln(1-(1-f_ret)^2) for α ∈ [0.40, 1.00] at
  large N (default N=500), using the smart-MC truncation derived in
  latex/smart_MC_validity.tex.

Two thresholds (β_net=1, derived in the latex writeup):
  • λ_var ≈ 0.34  — self-averaging threshold. Below it the heavy-tailed
    weight distribution makes σ_S/C ≥ 1 at any K, any N; the
    constant-floor approximation fails. Above it, σ_S/C → 0 exponentially
    in N. The α-grid starts at 0.40, safely above 0.34.
  • λ_w  ≈ 0.62  — floor-vs-live crossover (NOT a validity bound). Below
    α=0.62 the live patterns dominate the LSE; above, the constant
    floor C does. Basin geometry is preserved on both sides; the
    simulator computes H = -(1/β_net) log(C + p_live) exactly via
    max-shift LSE — the regimes differ only in analytical reasoning.

Grids:
  α-grid:  0.40 : 0.05 : 1.00              (13 values)
  T-grid:  per-α wedge, 30 points from
             T_MIN = 0.005   to   T_SAFETY · T_max(α, N, K)
           with  T_max(α,N,K) = (1 - φ_cut²)/φ_cut.
           T_max is monotone decreasing in α; e.g. at N=500, K=10⁴:
             α=0.40 → T_max ≈ 0.61   (top of grid ≈ 0.55)
             α=0.50 → T_max ≈ 0.43   (top ≈ 0.39)
             α=0.70 → T_max ≈ 0.25   (top ≈ 0.22)
             α=1.00 → T_max ≈ 0.15   (top ≈ 0.14)
           All comfortably above the physical T_crit(α).

Smart-MC scheme:
  Fixed N and budget K << M = exp(α N).  Live patterns ξ^2…ξ^K are
  drawn from the upper tail of the spherical density,
       (1+φ)/2 ~ Beta((N-1)/2,(N-1)/2)  conditioned on  φ > φ_cut,
  where  Pr(φ > φ_cut) = K / M.  The remaining (M-K) "passive" patterns
  are replaced by a constant C = (M-K)·E_f[exp(-β_net·N·(1-φ))] inside
  the log-sum-exp:
       H = -(1/β_net) ln( Σ_{μ=1}^K exp(-β_net·N·(1-φ_μ(x))) + C ).

Per-(α) wedge T-grid:
  T_max(α, N, K) = (1 - φ_cut²)/φ_cut       (smart-MC validity wall)
  T_grid(α) = range(T_min, 0.9·T_max(α), length = N_T_PER_ALPHA)

Output: basin_stab_LSE_smart_v18_N<N>_K<K>.csv
  columns: alpha, T, disorder, phi_a, phi_b, q12, phi_max_other
  preceded by a header line  '# N=… K=… betanet=…'
────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter
using SpecialFunctions
using QuadGK

# ──────────────── Precision Settings ────────────────
const USE_FLOAT16 = false
const F = USE_FLOAT16 ? Float16 : Float32

# ──────────────── Smart-MC parameters ────────────────
const N             = 500                    # spin count (fixed for the run)
const BUDGET_K      = 10_000                 # live-pattern budget
const betanet       = F(1.0)                 # LSE kernel inverse variance (default 1.0)

# ──────────────── MC parameters ────────────────
const MAX_N_TRIALS  = 256
const MIN_N_TRIALS  = 64
const N_EQ          = 2^15                   # 16384 equilibration steps (unmeasured)
const N_SAMP        = 2^13                   # 4096 sampling steps (measured)

# ──────────────── Sweep grids ────────────────
# α ∈ [0.40, 1.00]:  smart-MC heavy-tail self-averaging requires α > 0.34
#   (heaviness ≈ exp(0.17 N), √M = exp(αN/2)).
#   At N=200, σ_S/C ≈ exp((0.17-α/2)·N) ≤ 3e-3 across this band.
const alpha_vec     = collect(F(0.40):F(0.05):F(1.00))   # 13 values
const n_alpha       = length(alpha_vec)
const N_T_PER_ALPHA = 30
const T_MIN         = F(0.005)
const T_SAFETY      = F(0.90)                # cap T-grid at 0.9·T_max(α)

const TARGET_MEM_PER_CHUNK_GB = 40.0

# Disorder samples per α: light geometric ramp (cheap at high α anyway)
function n_trials_for_alpha(idx::Int)
    t = (idx - 1) / max(1, n_alpha - 1)
    trials = round(Int, MAX_N_TRIALS * (1 - t) + MIN_N_TRIALS * t)
    return max(MIN_N_TRIALS, trials)
end
const n_trials_vec   = [n_trials_for_alpha(i) for i in 1:n_alpha]
const n_disorder_vec = [n_trials_vec[i] ÷ 2 for i in 1:n_alpha]

# ──────────────── Theory: Beta upper-tail tools ─────────────────────────────
const beta_a = (N - 1)/2

# Log-survival P(X > x) for symmetric Beta(a,a); robust deep tail
function log_beta_sf(x::Float64, a::Float64)
    x <= 0 && return 0.0
    x >= 1 && return -Inf
    if x < 0.95
        _, q = beta_inc(a, a, x); q > 0 && return log(q)
    end
    u = 1 - x
    integrand(t) = t > 0 ? exp((a-1)*(log(t) + log1p(-u*t))) : 0.0
    val, _ = quadgk(integrand, 0.0, 1.0; rtol=1e-10)
    return a*log(u) + log(val) - logbeta(a, a)
end

# x such that log P(X > x) = log_p, in log-u space
function beta_isf_log(log_p::Float64, a::Float64)
    log_p >= 0 && return 0.0
    isinf(log_p) && log_p < 0 && return 1.0
    log_u = (log_p + log(a) + logbeta(a, a)) / a
    log_u = min(log_u, log(0.5))
    for _ in 1:60
        u = clamp(exp(log_u), 1e-300, 0.5); x = 1 - u
        lp = log_beta_sf(x, a); err = lp - log_p
        abs(err) < 1e-10 && break
        log_pdf = (a-1)*(log(x) + log(u)) - logbeta(a, a)
        dlog = exp(log_pdf + log(u) - lp); dlog < 1e-12 && (dlog = a)
        log_u -= err / dlog
    end
    return 1 - exp(log_u)
end

# φ_cut(α, N, K) such that Pr(φ > φ_cut) = K/M, M = exp(αN)
function phi_cut(α, N, K)
    log_p_keep = log(K) - α*N
    log_p_keep >= 0 && return -1.0           # K ≥ M : no truncation
    return 2*beta_isf_log(log_p_keep, Float64(beta_a)) - 1
end

# T_max from the bare validity bound  φ_eq(T) = φ_cut
T_max_bare(φc) = φc <= 0 ? Inf : (1 - φc^2)/φc

# Passive-sea constant  C = (M-K) · E_f[ exp(-β_net·N·(1-φ)) ]
# with f(φ) = Beta-on-(1+φ)/2.  In x = (1+φ)/2 coords:  1-φ = 2(1-x).
function passive_C(α, N, K, βn::Float64)
    M = exp(α*N)
    a = Float64(beta_a)
    integrand(x) = begin
        (x <= 0 || x >= 1) && return 0.0
        log_pdf = (a-1)*(log(x) + log1p(-x)) - logbeta(a, a)
        exp(-2*βn*N*(1-x) + log_pdf)
    end
    I, _ = quadgk(integrand, 0.0, 1.0; rtol=1e-10)
    return (M - K) * I
end

# ──────────────── Pattern generation ────────────────
# Generate K live patterns for ONE disorder realisation.
# Returns Matrix{F} of size (N, K).  Column 1 is ξ¹ (champion).
function generate_live_patterns(α::Float64, K::Int, N::Int)::Matrix{F}
    # 1. champion ξ¹ uniform on √N-sphere
    g = randn(N)
    ξ1 = sqrt(N) .* (g ./ norm(g))

    pats = Matrix{F}(undef, N, K)
    pats[:, 1] .= F.(ξ1)

    log_KM = log(K) - α*N                 # ≤ 0 ; log of Pr(φ > φ_cut)
    a = Float64(beta_a)

    if log_KM >= 0
        # K ≥ M : draw all patterns from the full sphere (degenerate case)
        for μ in 2:K
            g = randn(N); g .= sqrt(N) .* (g ./ norm(g))
            pats[:, μ] .= F.(g)
        end
        return pats
    end

    # Pre-compute pieces of ξ¹ for the orthogonal decomposition
    u1 = ξ1 ./ sqrt(N)                    # unit vector along ξ¹
    for μ in 2:K
        # 2. draw φ_μ from truncated Beta upper tail
        log_u = log(rand()) + log_KM      # uniform on (-∞, log_KM)
        x_μ  = beta_isf_log(log_u, a)     # corresponding upper quantile
        φ_μ  = 2*x_μ - 1                  # in (φ_cut, 1)

        # 3. orthogonal direction:  uniform on (N-2)-sphere ⊥ ξ¹
        g = randn(N)
        proj = dot(g, u1)
        g .-= proj .* u1
        g ./= norm(g)                     # unit, ⊥ ξ¹

        # 4. ξ_μ = φ_μ·ξ¹ + √(N(1-φ_μ²))·g
        pats[:, μ] .= F.(φ_μ .* ξ1 .+ sqrt(N*(1 - φ_μ^2)) .* g)
    end
    return pats
end

# ──────────────── Step size (temperature-dependent) ────────────────
function make_ss_gpu(T_grid::Vector{F}, N::Int)
    ss_cpu = F.(2.4 .* T_grid ./ sqrt(F(N)))
    return CuArray(reshape(ss_cpu, 1, length(T_grid), 1))
end

# ──────────────── LSE energy with passive-sea constant C ────────────────
# H = -(1/β_net) ln( Σ_μ exp(-β_net (N - x·ξ_μ)) + exp(logC) )
function compute_energy_lse_C!(E::CuVector{F}, x::CuArray{F,3},
                                patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                Nf::F, logC::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = -betanet * (Nf - overlap)         # arg_μ
    m_pat = maximum(overlap, dims=1)               # max over live patterns
    m = max.(m_pat, logC)                          # include the C term in the LSE
    @. overlap = exp(overlap - m)
    s_pat = sum(overlap, dims=1)
    s_total = s_pat .+ exp.(logC .- m)
    E .= vec(@. -(m + log(s_total)) / betanet)
    return nothing
end

# Max overlap with any non-target live pattern (excludes ξ¹)
function compute_phi_max_other!(phi_max_out::CuVector{F}, x::CuArray{F,3},
                                 patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                 Nf::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = overlap / Nf
    overlap[1, :, :] .= F(-1e30)                    # exclude champion
    mx = maximum(overlap, dims=1)
    phi_max_out .+= vec(mx)
    return nothing
end

# ──────────────── MC step ────────────────
function mc_step!(x, xp, E, Ep, pat, ov, β, ra, Nf, ss, logC, n_T_local)
    randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lse_C!(Ep, xp, pat, ov, Nf, logC)

    rand!(ra)
    acc = @. (ra < exp(-(β * (Ep - E))))
    n_dis = length(β) ÷ n_T_local
    a3 = reshape(acc, 1, n_T_local, n_dis)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

# Initialise each replica at x = ξ¹  (no perturbation; equilibration handles it)
function initialise_at_target!(x::CuArray{F,3}, target::CuArray{F,3}, n_T_local::Int)
    @views for j in 1:n_T_local
        x[:, j, :] .= target[:, 1, :]
    end
    return nothing
end

# ──────────────── CLI & Resume ────────────────
const FRESH_START = "--fresh" in ARGS
const csv_out     = @sprintf("basin_stab_LSE_smart_v18_N%d_K%d.csv", N, BUDGET_K)

function read_completed_alphas(csv_file::String)
    !isfile(csv_file) && return Set{String}()
    seen = Set{String}()
    first_data = true
    open(csv_file, "r") do f
        for line in eachline(f)
            isempty(line) && continue
            startswith(line, "#") && continue          # skip metadata
            if first_data
                first_data = false; continue           # skip header
            end
            push!(seen, String(split(line, ",")[1]))
        end
    end
    return seen
end

function sort_csv!(csv_file::String)
    !isfile(csv_file) && return
    lines = readlines(csv_file)
    isempty(lines) && return
    meta = String[]; header = ""; data = String[]
    first_data = true
    for l in lines
        if startswith(l, "#"); push!(meta, l); continue; end
        if first_data; header = l; first_data = false; continue; end
        isempty(l) && continue
        push!(data, l)
    end
    function sortkey(l)
        parts = split(l, ",")
        return (parse(Float64, parts[1]),
                parse(Float64, parts[2]),
                parse(Int,     parts[3]))
    end
    sort!(data, by=sortkey)
    tmp = csv_file * ".tmp"
    open(tmp, "w") do f
        for m in meta; println(f, m); end
        println(f, header)
        for l in data; println(f, l); end
    end
    mv(tmp, csv_file; force=true)
    return nothing
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("=" ^ 72)
    println("Smart-MC LSE Basin Stability  (v18)")
    @printf("  N = %d   K = %d   β_net = %s   precision = %d-bit\n",
            N, BUDGET_K, betanet, sizeof(F)*8)
    println("  α range: $(alpha_vec[1]) – $(alpha_vec[end])   step $(alpha_vec[2]-alpha_vec[1])")
    println("  Per-α wedge T-grid: $(N_T_PER_ALPHA) points, $T_MIN to $T_SAFETY·T_max(α)")
    @printf("  Trials: %d (α_min) → %d (α_max)   disorder = trials/2\n",
            MAX_N_TRIALS, MIN_N_TRIALS)
    println("  Equilibration: $N_EQ steps   Sampling: $N_SAMP steps")
    println("  Passive-sea constant C absorbed inside LSE log")
    println("=" ^ 72)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # Validity check: per-α φ_cut, T_max, T-grid, log C
    println("Computing validity envelope per α...")
    phi_cuts  = zeros(Float64, n_alpha)
    T_maxes   = zeros(Float64, n_alpha)
    T_grids   = Vector{Vector{F}}(undef, n_alpha)
    log_Cs    = zeros(Float64, n_alpha)
    for (i, α) in enumerate(alpha_vec)
        α64 = Float64(α)
        phi_cuts[i] = phi_cut(α64, N, BUDGET_K)
        T_maxes[i]  = T_max_bare(phi_cuts[i])
        T_hi        = min(F(T_SAFETY * T_maxes[i]), F(2.0))
        T_hi        = max(T_hi, T_MIN + F(1e-3))
        T_grids[i]  = collect(range(T_MIN, T_hi, length=N_T_PER_ALPHA))
        log_Cs[i]   = log(passive_C(α64, N, BUDGET_K, Float64(betanet)))
    end

    @printf("  φ_cut: %.4f (α=%.2f) … %.4f (α=%.2f)\n",
            phi_cuts[1], alpha_vec[1], phi_cuts[end], alpha_vec[end])
    @printf("  T_max: %.4f (α=%.2f) … %.4f (α=%.2f)\n",
            T_maxes[1], alpha_vec[1], T_maxes[end], alpha_vec[end])
    @printf("  log C: %.3e (α=%.2f) … %.3e (α=%.2f)\n",
            log_Cs[1], alpha_vec[1], log_Cs[end], alpha_vec[end])
    println()

    # Memory budget per α (single GPU chunk):  patterns dominate
    Nf = F(N)
    n_chains_max = maximum(n_disorder_vec) * N_T_PER_ALPHA
    mem_per_alpha = (N*BUDGET_K + N + 3*N*N_T_PER_ALPHA +
                     BUDGET_K*N_T_PER_ALPHA + 9*n_chains_max + N_T_PER_ALPHA) *
                    sizeof(F) * maximum(n_disorder_vec)
    @printf("Memory per α (upper bound): %.2f GB\n", mem_per_alpha/1e9)
    available_mem = CUDA.free_memory() * 0.85
    target_mem    = min(available_mem, TARGET_MEM_PER_CHUNK_GB * 1e9)
    chunk_size    = max(1, floor(Int, target_mem / mem_per_alpha))
    @printf("Available GPU memory: %.2f GB   chunk_size = %d α/chunk\n\n",
            available_mem/1e9, chunk_size)

    # Initialise / resume CSV
    if FRESH_START || !isfile(csv_out)
        open(csv_out, "w") do f
            @printf(f, "# N=%d  K=%d  betanet=%s  generated=%s\n",
                    N, BUDGET_K, betanet, string(now()))
            write(f, "alpha,T,disorder,phi_a,phi_b,q12,phi_max_other\n")
        end
        println(FRESH_START ? "Fresh start (--fresh)." : "No existing CSV found, starting fresh.")
    end
    completed = FRESH_START ? Set{String}() : read_completed_alphas(csv_out)
    pending_indices = Int[]
    for i in 1:n_alpha
        key = @sprintf("%.3f", alpha_vec[i])
        !(key in completed) && push!(pending_indices, i)
    end
    n_pending = length(pending_indices)
    if n_pending == 0
        println("All $n_alpha α values already present in CSV. Nothing to do.")
        sort_csv!(csv_out); return
    end
    @printf("Pending α: %d/%d.  ", n_pending, n_alpha)
    print("Indices: "); for i in pending_indices; @printf("%.2f ", alpha_vec[i]); end; println("\n")

    t_total_eq = 0.0; t_total_samp = 0.0

    for pend_start in 1:chunk_size:n_pending
        pend_end       = min(pend_start + chunk_size - 1, n_pending)
        chunk_indices  = pending_indices[pend_start:pend_end]
        n_chunk        = length(chunk_indices)

        println("\n" * "─"^72)
        @printf("Chunk: pending %d–%d of %d  (α = %s)\n", pend_start, pend_end, n_pending,
                join([@sprintf("%.2f", alpha_vec[i]) for i in chunk_indices], ", "))
        println("─"^72)

        # ── Allocate GPU memory ──
        pats_g    = Vector{CuArray{F,3}}(undef, n_chunk)
        tgts_g    = Vector{CuArray{F,3}}(undef, n_chunk)
        xa_g      = Vector{CuArray{F,3}}(undef, n_chunk)
        xb_g      = Vector{CuArray{F,3}}(undef, n_chunk)
        xp_g      = Vector{CuArray{F,3}}(undef, n_chunk)
        ov_g      = Vector{CuArray{F,3}}(undef, n_chunk)
        Ea_g      = Vector{CuVector{F}}(undef, n_chunk)
        Eb_g      = Vector{CuVector{F}}(undef, n_chunk)
        Ep_g      = Vector{CuVector{F}}(undef, n_chunk)
        phia_g    = Vector{CuVector{F}}(undef, n_chunk)
        phib_g    = Vector{CuVector{F}}(undef, n_chunk)
        qs_g      = Vector{CuVector{F}}(undef, n_chunk)
        phimax_g  = Vector{CuVector{F}}(undef, n_chunk)
        β_g       = Vector{CuVector{F}}(undef, n_chunk)
        ra_g      = Vector{CuVector{F}}(undef, n_chunk)
        ss_g      = Vector{CuArray{F,3}}(undef, n_chunk)
        nT_local  = Vector{Int}(undef, n_chunk)
        logC_loc  = Vector{F}(undef, n_chunk)

        for (li, gi) in enumerate(chunk_indices)
            α64    = Float64(alpha_vec[gi])
            n_dis  = n_disorder_vec[gi]
            T_grid = T_grids[gi]
            n_T_l  = length(T_grid)
            nT_local[li] = n_T_l
            logC_loc[li] = F(log_Cs[gi])
            n_chains     = n_T_l * n_dis

            # CPU pattern generation per disorder (uses CPU RNG with reproducible seed)
            Random.seed!(42 + 1000*gi)
            p_cpu = Array{F,3}(undef, N, BUDGET_K, n_dis)
            for d in 1:n_dis
                p_cpu[:, :, d] .= generate_live_patterns(α64, BUDGET_K, N)
            end

            pats_g[li]   = CuArray(p_cpu)
            tgts_g[li]   = CuArray(p_cpu[:, 1:1, :])
            xa_g[li]     = CUDA.zeros(F, N, n_T_l, n_dis)
            xb_g[li]     = CUDA.zeros(F, N, n_T_l, n_dis)
            xp_g[li]     = CUDA.zeros(F, N, n_T_l, n_dis)
            ov_g[li]     = CUDA.zeros(F, BUDGET_K, n_T_l, n_dis)
            Ea_g[li]     = CUDA.zeros(F, n_chains)
            Eb_g[li]     = CUDA.zeros(F, n_chains)
            Ep_g[li]     = CUDA.zeros(F, n_chains)
            phia_g[li]   = CUDA.zeros(F, n_chains)
            phib_g[li]   = CUDA.zeros(F, n_chains)
            qs_g[li]     = CUDA.zeros(F, n_chains)
            phimax_g[li] = CUDA.zeros(F, n_chains)
            β_g[li]      = CuVector{F}(repeat(F.(1 ./ T_grid), n_dis))
            ra_g[li]     = CUDA.zeros(F, n_chains)
            ss_g[li]     = make_ss_gpu(T_grid, N)
        end
        CUDA.synchronize()

        # ── Initialise both replicas at x = ξ¹ ──
        for li in 1:n_chunk
            initialise_at_target!(xa_g[li], tgts_g[li], nT_local[li])
            initialise_at_target!(xb_g[li], tgts_g[li], nT_local[li])
            compute_energy_lse_C!(Ea_g[li], xa_g[li], pats_g[li], ov_g[li], Nf, logC_loc[li])
            compute_energy_lse_C!(Eb_g[li], xb_g[li], pats_g[li], ov_g[li], Nf, logC_loc[li])
        end
        CUDA.synchronize()

        # ── Equilibration ──
        println("Equilibration ($N_EQ steps)...")
        t0 = time()
        prog = Progress(N_EQ, desc="Equilibration: ")
        for _ in 1:N_EQ
            for li in 1:n_chunk
                mc_step!(xa_g[li], xp_g[li], Ea_g[li], Ep_g[li],
                         pats_g[li], ov_g[li], β_g[li], ra_g[li],
                         Nf, ss_g[li], logC_loc[li], nT_local[li])
                mc_step!(xb_g[li], xp_g[li], Eb_g[li], Ep_g[li],
                         pats_g[li], ov_g[li], β_g[li], ra_g[li],
                         Nf, ss_g[li], logC_loc[li], nT_local[li])
            end
            next!(prog)
        end
        finish!(prog); CUDA.synchronize()
        t_eq = time() - t0; t_total_eq += t_eq
        @printf("  %.1f s (%.2f ms/step)\n", t_eq, 1000*t_eq/N_EQ)

        # ── Sampling ──
        for li in 1:n_chunk
            phia_g[li]   .= zero(F); phib_g[li]   .= zero(F)
            qs_g[li]     .= zero(F); phimax_g[li] .= zero(F)
        end
        println("Sampling ($N_SAMP steps)...")
        t0 = time()
        prog = Progress(N_SAMP, desc="Sampling: ")
        for _ in 1:N_SAMP
            for li in 1:n_chunk
                mc_step!(xa_g[li], xp_g[li], Ea_g[li], Ep_g[li],
                         pats_g[li], ov_g[li], β_g[li], ra_g[li],
                         Nf, ss_g[li], logC_loc[li], nT_local[li])
                mc_step!(xb_g[li], xp_g[li], Eb_g[li], Ep_g[li],
                         pats_g[li], ov_g[li], β_g[li], ra_g[li],
                         Nf, ss_g[li], logC_loc[li], nT_local[li])
                phia_g[li] .+= vec(sum(tgts_g[li] .* xa_g[li], dims=1)) ./ Nf
                phib_g[li] .+= vec(sum(tgts_g[li] .* xb_g[li], dims=1)) ./ Nf
                qs_g[li]   .+= vec(sum(xa_g[li] .* xb_g[li], dims=1)) ./ Nf
                compute_phi_max_other!(phimax_g[li], xa_g[li], pats_g[li],
                                        ov_g[li], Nf)
            end
            next!(prog)
        end
        finish!(prog); CUDA.synchronize()
        t_samp = time() - t0; t_total_samp += t_samp
        @printf("  %.1f s (%.2f ms/step)\n", t_samp, 1000*t_samp/N_SAMP)

        # ── Write CSV ──
        for (li, gi) in enumerate(chunk_indices)
            n_dis  = n_disorder_vec[gi]
            n_T_l  = nT_local[li]
            T_grid = T_grids[gi]
            phia_avg   = Array(phia_g[li])   ./ N_SAMP
            phib_avg   = Array(phib_g[li])   ./ N_SAMP
            q_avg      = Array(qs_g[li])     ./ N_SAMP
            phimax_avg = Array(phimax_g[li]) ./ N_SAMP
            phia_mat   = reshape(phia_avg,   n_T_l, n_dis)
            phib_mat   = reshape(phib_avg,   n_T_l, n_dis)
            q_mat      = reshape(q_avg,      n_T_l, n_dis)
            phimax_mat = reshape(phimax_avg, n_T_l, n_dis)
            open(csv_out, "a") do f
                for d in 1:n_dis, j in 1:n_T_l
                    @printf(f, "%.3f,%.5f,%d,%.6f,%.6f,%.6f,%.6f\n",
                            alpha_vec[gi], T_grid[j], d,
                            phia_mat[j, d], phib_mat[j, d],
                            q_mat[j, d], phimax_mat[j, d])
                end
            end
        end
        print("Sorting CSV... "); sort_csv!(csv_out); println("done.")

        # ── Free GPU memory ──
        pats_g = nothing; tgts_g = nothing
        xa_g = nothing; xb_g = nothing; xp_g = nothing
        ov_g = nothing; Ea_g = nothing; Eb_g = nothing
        Ep_g = nothing; phia_g = nothing; phib_g = nothing
        qs_g = nothing; phimax_g = nothing
        β_g = nothing; ra_g = nothing; ss_g = nothing
        GC.gc(); CUDA.reclaim()
    end

    println("\n" * "=" ^ 72)
    @printf("CSV saved: %s\n", csv_out)
    @printf("Total time: equilibration %.1f s,  sampling %.1f s\n", t_total_eq, t_total_samp)
    println("=" ^ 72)
end

# `now()` needed for the header timestamp
using Dates: now
main()
