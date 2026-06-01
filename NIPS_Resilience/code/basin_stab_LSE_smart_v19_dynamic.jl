#=
GPU-Accelerated Smart-MC LSE v19 — DYNAMIC LIVE-SET REGENERATION
────────────────────────────────────────────────────────────────────────
Usage:
  julia basin_stab_LSE_smart_v19_dynamic.jl           # resume
  julia basin_stab_LSE_smart_v19_dynamic.jl --fresh   # restart

Motivation: v18's static live set (K patterns near ξ¹) fails to reproduce
α_c^E(T) because the chain has no "champions of x" once it wanders away
from ξ¹.  Here we REGENERATE the K-1 dynamic competitors every N_REFRESH
MC steps, centred on the chain's CURRENT position rather than on ξ¹.
This restores honest MC's mechanism: at every x the chain has access to
the upper tail of the spherical density in x's frame.

Memory layout:
  tgts_g [N, n_dis]                — ξ¹ per disorder realisation
  pats_g [N, K-1, n_T, n_dis]      — dynamic competitors per (T, d) cell

Per disorder realisation, per (T, d) cell, the energy is
    H(x) = -(1/β_net) ln[ exp(-β_net·N·(1-φ_1(x)))                # ξ¹
                          + Σ_{μ=2..K} exp(-β_net·N·(1-φ_μ(x)))   # K-1 dyn
                          + C ]                                   # passive
where φ_1 = x·ξ¹/N and the K-1 dynamic competitors have overlap > φ_cut
with the cell's current x.  C absorbs the (M-K) remaining passive patterns
as in v18.

Output: basin_stab_LSE_smart_v19_N<N>_K<K>.csv  (same schema as v18)
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
using Dates: now

const USE_FLOAT16 = false
const F = USE_FLOAT16 ? Float16 : Float32

# ──────────────── Smart-MC parameters ────────────────
const N             = 500
const BUDGET_K      = 10_000               # includes ξ¹ as logical pattern 1
const N_DYN         = BUDGET_K - 1         # K-1 dynamic competitors per cell
const betanet       = F(1.0)
const N_REFRESH     = 100                  # MC steps between live-set regenerations

# ──────────────── MC parameters ────────────────
const MAX_N_TRIALS  = 32
const MIN_N_TRIALS  = 16
const N_EQ          = 2^15
const N_SAMP        = 2^13

const alpha_vec     = collect(F(0.40):F(0.05):F(1.00))
const n_alpha       = length(alpha_vec)
const N_T_PER_ALPHA = 20
const T_MIN         = F(0.005)
const T_SAFETY      = F(0.90)

const TARGET_MEM_PER_CHUNK_GB = 40.0

function n_trials_for_alpha(idx::Int)
    t = (idx - 1) / max(1, n_alpha - 1)
    trials = round(Int, MAX_N_TRIALS * (1 - t) + MIN_N_TRIALS * t)
    return max(MIN_N_TRIALS, trials)
end
const n_trials_vec   = [n_trials_for_alpha(i) for i in 1:n_alpha]
const n_disorder_vec = [n_trials_vec[i] ÷ 2 for i in 1:n_alpha]

# ──────────────── Beta upper-tail tools ─────────────────────────────
const beta_a = (N - 1) / 2

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

function phi_cut(α::Float64, K::Int)
    log_p_keep = log(K) - α * N
    log_p_keep >= 0 && return -1.0
    return 2 * beta_isf_log(log_p_keep, Float64(beta_a)) - 1
end

# Passive-sea constant
function passive_C(α::Float64, K::Int, βn::Float64)
    M = exp(α * N)
    a = Float64(beta_a)
    integrand(x) = begin
        (x <= 0 || x >= 1) && return 0.0
        log_pdf = (a-1)*(log(x) + log1p(-x)) - logbeta(a, a)
        exp(-2*βn*N*(1-x) + log_pdf)
    end
    I, _ = quadgk(integrand, 0.0, 1.0; rtol=1e-10)
    return (M - K) * I
end

# ──────────────── Pre-computed quantile table per α ─────────────────
# Sampling: log_arg = log(K/M) + log(rand());  x = quantile(log_arg)
# Table maps log_arg ∈ [log(K/M)-200, log(K/M)] → x_quantile (= 2φ_μ in [0,1])
struct QuantileTable
    log_arg_lo::Float64
    log_arg_hi::Float64
    n::Int
    x_grid::Vector{Float64}        # x_quantile values at evenly spaced log_arg
end

function build_quantile_table(α::Float64; n_grid::Int = 1000, tail_depth::Float64 = 200.0)
    log_arg_hi = log(BUDGET_K) - α * N
    log_arg_lo = log_arg_hi - tail_depth
    a = Float64(beta_a)
    log_args = range(log_arg_lo, log_arg_hi, length=n_grid)
    x_grid = [beta_isf_log(la, a) for la in log_args]
    return QuantileTable(log_arg_lo, log_arg_hi, n_grid, collect(x_grid))
end

@inline function sample_phi_from_table(table::QuantileTable, log_uniform::Float64, α::Float64)
    # log_arg = log(K/M) + log_uniform   where log_uniform = log(rand()) ∈ (-∞,0]
    log_arg = table.log_arg_hi + log_uniform
    log_arg = max(log_arg, table.log_arg_lo)
    # Linear interp in log_arg → x
    frac = (log_arg - table.log_arg_lo) / (table.log_arg_hi - table.log_arg_lo)
    idx_f = frac * (table.n - 1) + 1
    idx_lo = clamp(floor(Int, idx_f), 1, table.n - 1)
    w = idx_f - idx_lo
    x = table.x_grid[idx_lo] * (1 - w) + table.x_grid[idx_lo + 1] * w
    return 2 * x - 1                      # → φ
end

# ──────────────── Pattern generation (CPU) ──────────────────────────
# Generate K-1 dynamic competitors around the reference vector x_ref.
# x_ref is on the √N-sphere. Output: Matrix{F} of size (N, K-1).
function generate_competitors!(out::AbstractMatrix{F}, x_ref::AbstractVector{<:Real},
                                table::QuantileTable, α::Float64)
    K_dyn = size(out, 2)
    u_ref = x_ref ./ sqrt(N)
    g = zeros(N)
    for μ in 1:K_dyn
        log_u = log(rand())
        φ_μ  = sample_phi_from_table(table, log_u, α)
        # Random perpendicular direction
        randn!(g)
        proj = dot(g, u_ref)
        g .-= proj .* u_ref
        nrm = norm(g)
        nrm < 1e-12 && (randn!(g); g .-= dot(g, u_ref) .* u_ref; nrm = norm(g))
        g ./= nrm
        # ξ_μ = φ_μ · x_ref + √(N(1-φ_μ²)) · g
        @inbounds for i in 1:N
            out[i, μ] = F(φ_μ * x_ref[i] + sqrt(N * (1 - φ_μ^2)) * g[i])
        end
    end
    return out
end

# Initial setup: generate ξ¹ and initial pattern set per cell (around ξ¹).
function initial_patterns(α::Float64, n_dis::Int, n_T_l::Int, table::QuantileTable)
    tgt_cpu = Array{F,2}(undef, N, n_dis)
    pat_cpu = Array{F,4}(undef, N, N_DYN, n_T_l, n_dis)
    g = zeros(N)
    for d in 1:n_dis
        randn!(g)
        g .*= sqrt(N) / norm(g)
        tgt_cpu[:, d] .= F.(g)
    end
    for d in 1:n_dis, j in 1:n_T_l
        # initial reference = ξ¹ for this disorder
        x_ref = Float64.(tgt_cpu[:, d])
        view_p = view(pat_cpu, :, :, j, d)
        generate_competitors!(view_p, x_ref, table, α)
    end
    return tgt_cpu, pat_cpu
end

# Regenerate competitors using CURRENT chain positions x_chain (per cell).
function regenerate_patterns!(pat_cpu::Array{F,4}, x_chain_cpu::Array{F,3},
                               table::QuantileTable, α::Float64)
    n_T_l = size(pat_cpu, 3); n_dis = size(pat_cpu, 4)
    for d in 1:n_dis, j in 1:n_T_l
        x_ref = Float64.(view(x_chain_cpu, :, j, d))
        view_p = view(pat_cpu, :, :, j, d)
        generate_competitors!(view_p, x_ref, table, α)
    end
    return pat_cpu
end

# ──────────────── Step size ────────────────
function make_ss_gpu(T_grid::Vector{F}, N::Int)
    ss_cpu = F.(2.4 .* T_grid ./ sqrt(F(N)))
    return CuArray(reshape(ss_cpu, 1, length(T_grid), 1))
end

# ──────────────── LSE energy with ξ¹ + K-1 dynamic patterns + C ─────
# pats_g  : [N, K-1, n_T, n_dis]
# tgts_g  : [N, 1,   n_dis]      (ξ¹ per disorder; broadcast over T)
# x       : [N, n_T, n_dis]
# overlap : [K-1, n_T, n_dis]    (workspace for K-1 competitors)
function compute_energy_lse_dyn!(E::CuVector{F}, x::CuArray{F,3},
                                  tgts::CuArray{F,3}, pats::CuArray{F,4},
                                  overlap::CuArray{F,3}, Nf::F, logC::F)
    # Champion ξ¹: overlap per (T, d) = sum_i tgts[i, 1, d] * x[i, T, d]
    # produces shape (1, n_T, n_dis); we want exponent -β_net*(Nf - that)
    ov1 = sum(tgts .* x, dims=1)                  # (1, n_T, n_dis)
    arg1 = @. -betanet * (Nf - ov1)               # (1, n_T, n_dis)
    arg1_flat = vec(arg1)                         # n_chains

    # Dynamic competitors: per cell GEMM pats[:,:,j,d]^T * x[:,j,d]
    n_T_l = size(x, 2); n_dis = size(x, 3); K_dyn = size(pats, 2)
    pats_r = reshape(pats, N, K_dyn, n_T_l * n_dis)
    x_r    = reshape(x,    N, 1,     n_T_l * n_dis)
    ov_r   = reshape(overlap, K_dyn, 1, n_T_l * n_dis)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), pats_r, x_r, zero(F), ov_r)
    @. overlap = -betanet * (Nf - overlap)        # arg per dynamic pattern

    # Combined max-shift across (champion, dyn patterns, log C)
    m_dyn  = maximum(overlap, dims=1)             # (1, n_T, n_dis)
    m_pair = max.(m_dyn, arg1)                    # include champion
    m      = max.(m_pair, logC)                   # include passive floor

    # Sum: champion + Σ dynamic + C, all shifted by m
    @. overlap = exp(overlap - m)
    s_dyn  = sum(overlap, dims=1)                 # (1, n_T, n_dis)
    s_tot  = s_dyn .+ exp.(arg1 .- m) .+ exp.(logC .- m)
    E .= vec(@. -(m + log(s_tot)) / betanet)
    return nothing
end

# Max overlap with K-1 dynamic competitors (no champion exclusion needed)
function compute_phi_max_other_dyn!(phi_max_out::CuVector{F}, x::CuArray{F,3},
                                     pats::CuArray{F,4}, overlap::CuArray{F,3}, Nf::F)
    n_T_l = size(x, 2); n_dis = size(x, 3); K_dyn = size(pats, 2)
    pats_r = reshape(pats, N, K_dyn, n_T_l * n_dis)
    x_r    = reshape(x,    N, 1,     n_T_l * n_dis)
    ov_r   = reshape(overlap, K_dyn, 1, n_T_l * n_dis)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), pats_r, x_r, zero(F), ov_r)
    @. overlap = overlap / Nf
    mx = maximum(overlap, dims=1)
    phi_max_out .+= vec(mx)
    return nothing
end

# ──────────────── MC step ────────────────
function mc_step!(x, xp, E, Ep, tgts, pats, ov, β, ra, Nf, ss, logC, n_T_l)
    randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm

    compute_energy_lse_dyn!(Ep, xp, tgts, pats, ov, Nf, logC)

    rand!(ra)
    acc = @. (ra < exp(-(β * (Ep - E))))
    n_dis = length(β) ÷ n_T_l
    a3 = reshape(acc, 1, n_T_l, n_dis)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

function initialise_at_target!(x::CuArray{F,3}, tgts::CuArray{F,3}, n_T_l::Int)
    @views for j in 1:n_T_l
        x[:, j, :] .= tgts[:, 1, :]
    end
    return nothing
end

# ──────────────── CLI & Resume ────────────────
const FRESH_START = "--fresh" in ARGS
const csv_out     = @sprintf("basin_stab_LSE_smart_v19_N%d_K%d.csv", N, BUDGET_K)

function read_completed_alphas(csv_file::String)
    !isfile(csv_file) && return Set{String}()
    seen = Set{String}()
    first_data = true
    open(csv_file, "r") do f
        for line in eachline(f)
            isempty(line) && continue
            startswith(line, "#") && continue
            if first_data; first_data = false; continue; end
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
    println("Smart-MC LSE v19 — DYNAMIC live-set regeneration")
    @printf("  N=%d  K=%d (1 champion + %d dynamic)  β_net=%s  precision=%d-bit\n",
            N, BUDGET_K, N_DYN, betanet, sizeof(F)*8)
    println("  Regeneration every $(N_REFRESH) MC steps")
    println("  α range: $(alpha_vec[1]) – $(alpha_vec[end])")
    @printf("  Trials: %d (α_min) → %d (α_max)   disorder = trials/2\n",
            MAX_N_TRIALS, MIN_N_TRIALS)
    println("  Eq: $N_EQ   Samp: $N_SAMP")
    println("=" ^ 72)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    # Pre-compute quantile tables, validity envelope, log C per α
    println("Building quantile tables and validity envelopes...")
    tables   = QuantileTable[]
    phi_cuts = zeros(Float64, n_alpha)
    T_maxes  = zeros(Float64, n_alpha)
    T_grids  = Vector{Vector{F}}(undef, n_alpha)
    log_Cs   = zeros(Float64, n_alpha)
    for (i, α) in enumerate(alpha_vec)
        α64 = Float64(α)
        push!(tables, build_quantile_table(α64; n_grid=500, tail_depth=200.0))
        phi_cuts[i] = phi_cut(α64, BUDGET_K)
        T_maxes[i]  = phi_cuts[i] <= 0 ? 2.0 : (1 - phi_cuts[i]^2)/phi_cuts[i]
        T_hi        = min(F(T_SAFETY * T_maxes[i]), F(2.0))
        T_hi        = max(T_hi, T_MIN + F(1e-3))
        T_grids[i]  = collect(range(T_MIN, T_hi, length=N_T_PER_ALPHA))
        log_Cs[i]   = log(passive_C(α64, BUDGET_K, Float64(betanet)))
    end
    @printf("  φ_cut: %.4f → %.4f\n", phi_cuts[1], phi_cuts[end])
    @printf("  T_max: %.4f → %.4f\n", T_maxes[1], T_maxes[end])
    @printf("  log C: %.2e → %.2e\n", log_Cs[1], log_Cs[end])
    println()

    Nf = F(N)
    if FRESH_START || !isfile(csv_out)
        open(csv_out, "w") do f
            @printf(f, "# N=%d  K=%d  N_REFRESH=%d  betanet=%s  generated=%s\n",
                    N, BUDGET_K, N_REFRESH, betanet, string(now()))
            write(f, "alpha,T,disorder,phi_a,phi_b,q12,phi_max_other\n")
        end
        println(FRESH_START ? "Fresh start (--fresh)." : "No existing CSV; starting fresh.")
    end
    completed = FRESH_START ? Set{String}() : read_completed_alphas(csv_out)
    pending_indices = Int[]
    for i in 1:n_alpha
        key = @sprintf("%.3f", alpha_vec[i])
        !(key in completed) && push!(pending_indices, i)
    end
    n_pending = length(pending_indices)
    if n_pending == 0
        println("All α already present in CSV.  Nothing to do."); sort_csv!(csv_out); return
    end
    @printf("Pending α: %d/%d\n\n", n_pending, n_alpha)

    t_total_eq = 0.0; t_total_samp = 0.0; t_total_regen = 0.0

    # Process one α at a time
    for gi in pending_indices
        α      = alpha_vec[gi]
        α64    = Float64(α)
        n_dis  = n_disorder_vec[gi]
        T_grid = T_grids[gi]
        n_T_l  = length(T_grid)
        n_chains = n_T_l * n_dis
        logC   = F(log_Cs[gi])
        table  = tables[gi]

        # ── Memory check ──
        pat_mem = N * N_DYN * n_T_l * n_dis * sizeof(F)
        println("─"^72)
        @printf("α=%.2f   n_dis=%d  n_T=%d  pat_mem=%.2f GB\n",
                α, n_dis, n_T_l, pat_mem/1e9)
        println("─"^72)

        # ── Initial pattern generation ──
        print("  Initial pattern generation... ")
        t0 = time()
        Random.seed!(42 + 1000*gi)
        tgt_cpu, pat_cpu = initial_patterns(α64, n_dis, n_T_l, table)
        @printf("%.1f s\n", time()-t0)

        # ── Move to GPU and allocate ──
        print("  GPU allocation/transfer... "); t0 = time()
        tgts_g = CuArray(reshape(tgt_cpu, N, 1, n_dis))
        pats_g = CuArray(pat_cpu)
        xa_g   = CUDA.zeros(F, N, n_T_l, n_dis)
        xb_g   = CUDA.zeros(F, N, n_T_l, n_dis)
        xp_g   = CUDA.zeros(F, N, n_T_l, n_dis)
        ov_g   = CUDA.zeros(F, N_DYN, n_T_l, n_dis)
        Ea_g, Eb_g, Ep_g = CUDA.zeros(F, n_chains), CUDA.zeros(F, n_chains), CUDA.zeros(F, n_chains)
        phia_g, phib_g    = CUDA.zeros(F, n_chains), CUDA.zeros(F, n_chains)
        qs_g, phimax_g    = CUDA.zeros(F, n_chains), CUDA.zeros(F, n_chains)
        β_g  = CuVector{F}(repeat(F.(1 ./ T_grid), n_dis))
        ra_g = CUDA.zeros(F, n_chains)
        ss_g = make_ss_gpu(T_grid, N)
        CUDA.synchronize(); @printf("%.1f s\n", time()-t0)

        # ── Initialise replicas at ξ¹ ──
        initialise_at_target!(xa_g, tgts_g, n_T_l)
        initialise_at_target!(xb_g, tgts_g, n_T_l)
        compute_energy_lse_dyn!(Ea_g, xa_g, tgts_g, pats_g, ov_g, Nf, logC)
        compute_energy_lse_dyn!(Eb_g, xb_g, tgts_g, pats_g, ov_g, Nf, logC)
        CUDA.synchronize()

        # ── Equilibration with dynamic regeneration ──
        println("  Equilibration ($N_EQ steps, refresh every $N_REFRESH)...")
        t0 = time(); t_regen_α = 0.0
        prog = Progress(N_EQ, desc="    eq: ")
        for step in 1:N_EQ
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, tgts_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, logC, n_T_l)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, tgts_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, logC, n_T_l)
            if step % N_REFRESH == 0
                tr0 = time()
                xa_cpu = Array(xa_g)         # reference = replica a's current position
                regenerate_patterns!(pat_cpu, xa_cpu, table, α64)
                copyto!(pats_g, pat_cpu)
                compute_energy_lse_dyn!(Ea_g, xa_g, tgts_g, pats_g, ov_g, Nf, logC)
                compute_energy_lse_dyn!(Eb_g, xb_g, tgts_g, pats_g, ov_g, Nf, logC)
                t_regen_α += time() - tr0
            end
            next!(prog)
        end
        finish!(prog); CUDA.synchronize()
        t_eq = time() - t0; t_total_eq += t_eq; t_total_regen += t_regen_α
        @printf("    %.1f s  (regen overhead %.1f s = %.0f%%)\n",
                t_eq, t_regen_α, 100*t_regen_α/t_eq)

        # ── Sampling with dynamic regeneration ──
        phia_g .= zero(F); phib_g .= zero(F); qs_g .= zero(F); phimax_g .= zero(F)
        println("  Sampling ($N_SAMP steps, refresh every $N_REFRESH)...")
        t0 = time(); t_regen_α = 0.0
        prog = Progress(N_SAMP, desc="    samp: ")
        for step in 1:N_SAMP
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, tgts_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, logC, n_T_l)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, tgts_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, logC, n_T_l)
            phia_g .+= vec(sum(tgts_g .* xa_g, dims=1)) ./ Nf
            phib_g .+= vec(sum(tgts_g .* xb_g, dims=1)) ./ Nf
            qs_g   .+= vec(sum(xa_g .* xb_g, dims=1)) ./ Nf
            compute_phi_max_other_dyn!(phimax_g, xa_g, pats_g, ov_g, Nf)
            if step % N_REFRESH == 0
                tr0 = time()
                xa_cpu = Array(xa_g)
                regenerate_patterns!(pat_cpu, xa_cpu, table, α64)
                copyto!(pats_g, pat_cpu)
                compute_energy_lse_dyn!(Ea_g, xa_g, tgts_g, pats_g, ov_g, Nf, logC)
                compute_energy_lse_dyn!(Eb_g, xb_g, tgts_g, pats_g, ov_g, Nf, logC)
                t_regen_α += time() - tr0
            end
            next!(prog)
        end
        finish!(prog); CUDA.synchronize()
        t_samp = time() - t0; t_total_samp += t_samp; t_total_regen += t_regen_α
        @printf("    %.1f s  (regen overhead %.1f s = %.0f%%)\n",
                t_samp, t_regen_α, 100*t_regen_α/t_samp)

        # ── Write CSV ──
        phia_avg   = Array(phia_g)   ./ N_SAMP
        phib_avg   = Array(phib_g)   ./ N_SAMP
        q_avg      = Array(qs_g)     ./ N_SAMP
        phimax_avg = Array(phimax_g) ./ N_SAMP
        phia_mat   = reshape(phia_avg,   n_T_l, n_dis)
        phib_mat   = reshape(phib_avg,   n_T_l, n_dis)
        q_mat      = reshape(q_avg,      n_T_l, n_dis)
        phimax_mat = reshape(phimax_avg, n_T_l, n_dis)
        open(csv_out, "a") do f
            for d in 1:n_dis, j in 1:n_T_l
                @printf(f, "%.3f,%.5f,%d,%.6f,%.6f,%.6f,%.6f\n",
                        α, T_grid[j], d,
                        phia_mat[j, d], phib_mat[j, d],
                        q_mat[j, d], phimax_mat[j, d])
            end
        end
        print("  Sorting CSV... "); sort_csv!(csv_out); println("done.")

        tgts_g=nothing; pats_g=nothing; xa_g=nothing; xb_g=nothing; xp_g=nothing
        ov_g=nothing; Ea_g=nothing; Eb_g=nothing; Ep_g=nothing
        phia_g=nothing; phib_g=nothing; qs_g=nothing; phimax_g=nothing
        β_g=nothing; ra_g=nothing; ss_g=nothing; pat_cpu=nothing; tgt_cpu=nothing
        GC.gc(); CUDA.reclaim()
    end

    println("\n" * "=" ^ 72)
    @printf("CSV saved: %s\n", csv_out)
    @printf("Total: eq %.1f s,  samp %.1f s,  regen %.1f s\n",
            t_total_eq, t_total_samp, t_total_regen)
    println("=" ^ 72)
end

main()
