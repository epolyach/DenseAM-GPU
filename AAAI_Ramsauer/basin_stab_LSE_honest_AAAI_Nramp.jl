#=
GPU-Accelerated HONEST LSE Basin Stability — AAAI 2027  ·  N(α)-RAMP
────────────────────────────────────────────────────────────────────────────────
Ground-truth companion to basin_stab_LSE_semismart_AAAI_Nramp.jl. No truncation:
all M = ⌈exp(αN)⌉ patterns are sampled explicitly per disorder. Same N(α) ramp
so the (α, N) grid matches the semismart Nramp cell-by-cell — directly tells
whether the semismart static-anchor bias is what pushes the boundary above the
red curve at large N, or whether the theory is the one to revise.

Scheme
  • M_TARGET = 4.4·10⁷         (≈ exp(0.70·25))
  • N(α)     = round(log(M_TARGET) / α)     (88 → 25 across α=0.20…0.70)
  • Disorder chunking: target N_DIS_TARGET per α. Chunk size = MEM_BUDGET_GB /
    per-disorder bytes (often 1 disorder per chunk at small α — patterns
    alone are ≈ N·M·4 ≈ 15 GB at α=0.20). Loop chunks until target met.
  • Resume tracks max disorder index per α.

Same energy / MC step / initialisation as basin_stab_LSE_honest_AAAI.jl.
CSV schema matches honest_AAAI so plot_LSE_AAAI_heatmap.jl loads it as a new
priority source.

Usage
  julia basin_stab_LSE_honest_AAAI_Nramp.jl              # resume
  julia basin_stab_LSE_honest_AAAI_Nramp.jl --fresh      # overwrite

Output: basin_stab_LSE_honest_AAAI_Nramp.csv
────────────────────────────────────────────────────────────────────────────────
=#

using CUDA
using Random
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter
using Dates: now

const F = Float32

# ──────────────── N(α) ramp ────────────────
const M_TARGET      = 4.4e7
const N_FLOOR       = 12
const betanet       = F(1.0)
const N_EQ          = 2^15
const N_SAMP        = 2^13
const N_DIS_TARGET  = 32          # disorder samples per α — honest is expensive at large N
const MEM_BUDGET_GB = 40.0

# const alpha_vec = collect(F(0.20):F(0.01):F(0.70))
const alpha_vec = F[0.30]
const T_vec     = collect(F(0.005):F(0.01):F(0.485))
const n_alpha   = length(alpha_vec)
const n_T       = length(T_vec)

N_for_alpha(α::Real) = max(N_FLOOR, round(Int, log(M_TARGET) / Float64(α)))

# ──────────────── Memory and chunking ────────────────
function mem_per_disorder_bytes(N::Int, M::Int)
    elems = Float64(N)*M + 3*Float64(N)*n_T + Float64(M)*n_T + 9*n_T + n_T
    return elems * sizeof(F)
end

function pick_chunk_size(N::Int, M::Int)
    per = mem_per_disorder_bytes(N, M)
    by_mem = floor(Int, MEM_BUDGET_GB * 1e9 / per)
    return clamp(by_mem, 1, N_DIS_TARGET)
end

# ──────────────── LSE energy and MC step ────────────────
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

function compute_phi_max_other!(phi_max_out::CuVector{F}, x::CuArray{F,3},
                                 patterns::CuArray{F,3}, overlap::CuArray{F,3},
                                 Nf::F)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', one(F), patterns, x, zero(F), overlap)
    @. overlap = overlap / Nf
    overlap[1, :, :] .= F(-1e30)
    mx = maximum(overlap, dims=1)
    phi_max_out .+= vec(mx)
    return nothing
end

function mc_step!(x, xp, E, Ep, pat, ov, β, ra, Nf, ss, n_dis_local::Int)
    randn!(xp)
    @. xp = x + ss * xp
    nrm = sqrt.(sum(xp .^ 2, dims=1))
    @. xp = sqrt(Nf) * xp / nrm
    compute_energy_lse!(Ep, xp, pat, ov, Nf)
    rand!(ra)
    acc = @. (ra < exp(-(β * (Ep - E))))
    a3 = reshape(acc, 1, n_T, n_dis_local)
    @. x = ifelse(a3, xp, x)
    @. E = ifelse(acc, Ep, E)
    return nothing
end

function make_ss_gpu(N::Int)
    Nf = F(N)
    ss_cpu = F.(2.4 .* T_vec ./ sqrt(Nf))
    return CuArray(reshape(ss_cpu, 1, n_T, 1))
end

function initialise_at_target!(x::CuArray{F,3}, target::CuArray{F,3})
    @views for j in 1:n_T
        x[:, j, :] .= target[:, 1, :]
    end
    return nothing
end

# ──────────────── CLI & Resume ────────────────
const FRESH_START = "--fresh" in ARGS
const csv_out     = "basin_stab_LSE_honest_AAAI_Nramp.csv"

function read_disorder_progress(csv_file::String)
    !isfile(csv_file) && return Dict{String,Int}()
    counts = Dict{String,Int}()
    first_data = true
    open(csv_file, "r") do f
        for line in eachline(f)
            isempty(line) && continue
            startswith(line, "#") && continue
            if first_data; first_data = false; continue; end
            parts = split(line, ",")
            αkey  = String(parts[1])
            dis   = parse(Int, parts[4])      # disorder column index in honest schema
            counts[αkey] = max(get(counts, αkey, 0), dis)
        end
    end
    return counts
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
                parse(Int,     parts[4]))
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

# ──────────────── Per-α driver ────────────────
function run_alpha!(α::Float64, N::Int, M::Int, dis_start::Int, dis_target::Int, alpha_idx::Int)
    Nf = F(N)
    n_dis_chunk = pick_chunk_size(N, M)
    dis_done = dis_start
    chunk_idx = 0
    while dis_done < dis_target
        chunk_idx += 1
        n_dis = min(n_dis_chunk, dis_target - dis_done)
        println("─"^76)
        @printf("α=%.3f  N=%d  M=%d  chunk %d  n_dis=%d  (disorders %d..%d / %d)\n",
                α, N, M, chunk_idx, n_dis, dis_done+1, dis_done+n_dis, dis_target)

        Random.seed!(42 + 1000*alpha_idx + 7919*dis_done)
        print("  Generating patterns ... "); t0 = time()
        p_cpu = randn(F, N, M, n_dis)
        for d in 1:n_dis, j in 1:M
            c = @view p_cpu[:, j, d]
            c .*= sqrt(Nf) / norm(c)
        end
        @printf("%.1f s\n", time()-t0)

        n_chains = n_T * n_dis
        print("  GPU alloc ... "); t0 = time()
        pats_g    = CuArray(p_cpu)
        tgts_g    = CuArray(p_cpu[:, 1:1, :])
        xa_g      = CUDA.zeros(F, N, n_T, n_dis)
        xb_g      = CUDA.zeros(F, N, n_T, n_dis)
        xp_g      = CUDA.zeros(F, N, n_T, n_dis)
        ov_g      = CUDA.zeros(F, M, n_T, n_dis)
        Ea_g      = CUDA.zeros(F, n_chains)
        Eb_g      = CUDA.zeros(F, n_chains)
        Ep_g      = CUDA.zeros(F, n_chains)
        phia_g    = CUDA.zeros(F, n_chains)
        phib_g    = CUDA.zeros(F, n_chains)
        qs_g      = CUDA.zeros(F, n_chains)
        phimax_g  = CUDA.zeros(F, n_chains)
        β_g       = CuVector{F}(repeat(F.(1 ./ T_vec), n_dis))
        ra_g      = CUDA.zeros(F, n_chains)
        ss_g      = make_ss_gpu(N)
        p_cpu = nothing; GC.gc(); CUDA.synchronize()
        @printf("%.1f s\n", time()-t0)

        initialise_at_target!(xa_g, tgts_g)
        initialise_at_target!(xb_g, tgts_g)
        compute_energy_lse!(Ea_g, xa_g, pats_g, ov_g, Nf)
        compute_energy_lse!(Eb_g, xb_g, pats_g, ov_g, Nf)
        CUDA.synchronize()

        println("  Equilibration ($N_EQ steps)…"); t0 = time()
        prog = Progress(N_EQ, desc="    eq: ")
        for _ in 1:N_EQ
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis)
            next!(prog)
        end
        finish!(prog); CUDA.synchronize(); @printf("    %.1f s\n", time()-t0)

        phia_g .= zero(F); phib_g .= zero(F); qs_g .= zero(F); phimax_g .= zero(F)
        println("  Sampling ($N_SAMP steps)…"); t0 = time()
        prog = Progress(N_SAMP, desc="    samp: ")
        for _ in 1:N_SAMP
            mc_step!(xa_g, xp_g, Ea_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis)
            mc_step!(xb_g, xp_g, Eb_g, Ep_g, pats_g, ov_g, β_g, ra_g, Nf, ss_g, n_dis)
            phia_g .+= vec(sum(tgts_g .* xa_g, dims=1)) ./ Nf
            phib_g .+= vec(sum(tgts_g .* xb_g, dims=1)) ./ Nf
            qs_g   .+= vec(sum(xa_g .* xb_g, dims=1)) ./ Nf
            compute_phi_max_other!(phimax_g, xa_g, pats_g, ov_g, Nf)
            next!(prog)
        end
        finish!(prog); CUDA.synchronize(); @printf("    %.1f s\n", time()-t0)

        phia_avg   = Array(phia_g)   ./ N_SAMP
        phib_avg   = Array(phib_g)   ./ N_SAMP
        q_avg      = Array(qs_g)     ./ N_SAMP
        phimax_avg = Array(phimax_g) ./ N_SAMP
        phia_mat   = reshape(phia_avg,   n_T, n_dis)
        phib_mat   = reshape(phib_avg,   n_T, n_dis)
        q_mat      = reshape(q_avg,      n_T, n_dis)
        phimax_mat = reshape(phimax_avg, n_T, n_dis)
        open(csv_out, "a") do f
            for d in 1:n_dis, j in 1:n_T
                @printf(f, "%.3f,%.5f,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                        α, T_vec[j], N, dis_done + d,
                        phia_mat[j, d], phib_mat[j, d],
                        q_mat[j, d], phimax_mat[j, d])
            end
        end

        pats_g = nothing; tgts_g = nothing
        xa_g = nothing; xb_g = nothing; xp_g = nothing
        ov_g = nothing; Ea_g = nothing; Eb_g = nothing; Ep_g = nothing
        phia_g = nothing; phib_g = nothing; qs_g = nothing; phimax_g = nothing
        β_g = nothing; ra_g = nothing; ss_g = nothing
        GC.gc(); CUDA.reclaim()

        dis_done += n_dis
    end
end

# ──────────────── Main ────────────────
function main()
    !CUDA.functional() && error("CUDA not available")

    println("="^76)
    println("HONEST LSE Basin Stability — AAAI 2027  ·  N(α)-RAMP")
    @printf("  M_TARGET = %.1e   N(α) = round(log(M)/α)   anchor: α=0.70 → N=25\n", M_TARGET)
    @printf("  N_FLOOR = %d   N_DIS_TARGET = %d   MEM_BUDGET = %.1f GB\n",
            N_FLOOR, N_DIS_TARGET, MEM_BUDGET_GB)
    @printf("  α grid: %.2f : %.2f : %.2f  (%d values)\n",
            alpha_vec[1], alpha_vec[2]-alpha_vec[1], alpha_vec[end], n_alpha)
    @printf("  T grid: %.4f : %.4f  (%d points)\n", T_vec[1], T_vec[end], n_T)
    @printf("  MC: %d eq + %d samp\n", N_EQ, N_SAMP)
    println("="^76)
    dev = CUDA.device()
    @printf("GPU: %s (%.1f GB)\n\n", CUDA.name(dev), CUDA.totalmem(dev)/1e9)

    plan = Vector{NamedTuple}(undef, n_alpha)
    println("Per-α plan (N, M, chunk size, est GB per disorder):")
    for i in 1:n_alpha
        α  = Float64(alpha_vec[i])
        N  = N_for_alpha(α)
        M  = round(Int, exp(α * N))
        ch = pick_chunk_size(N, M)
        per_gb = mem_per_disorder_bytes(N, M) / 1e9
        plan[i] = (alpha_idx=i, α=α, N=N, M=M, chunk=ch, per_gb=per_gb)
        @printf("  α=%.2f  N=%-3d  M=%-10d  chunk=%-2d  %.1f GB/dis\n",
                α, N, M, ch, per_gb)
    end
    println()

    if FRESH_START || !isfile(csv_out)
        open(csv_out, "w") do f
            @printf(f, "# generator=basin_stab_LSE_honest_AAAI_Nramp.jl  M_TARGET=%.1e  N_FLOOR=%d  betanet=%s\n",
                    M_TARGET, N_FLOOR, betanet)
            @printf(f, "# N_EQ=%d  N_SAMP=%d  N_DIS_TARGET=%d  MEM_BUDGET_GB=%.1f  generated=%s\n",
                    N_EQ, N_SAMP, N_DIS_TARGET, MEM_BUDGET_GB, string(now()))
            write(f, "alpha,T,N_used,disorder,phi_a,phi_b,q12,phi_max_other\n")
        end
        println(FRESH_START ? "Fresh start (--fresh)." : "No CSV — starting fresh.")
    end
    progress = FRESH_START ? Dict{String,Int}() : read_disorder_progress(csv_out)

    for p in plan
        αkey = @sprintf("%.3f", p.α)
        dis_done = get(progress, αkey, 0)
        if dis_done >= N_DIS_TARGET
            @printf("α=%.3f already has %d ≥ %d disorders, skipping.\n", p.α, dis_done, N_DIS_TARGET)
            continue
        end
        run_alpha!(p.α, p.N, p.M, dis_done, N_DIS_TARGET, p.alpha_idx)
        print("  Sorting CSV… "); sort_csv!(csv_out); println("done.")
    end

    println("\n" * "="^76)
    @printf("CSV: %s\n", csv_out)
    println("="^76)
end

main()
