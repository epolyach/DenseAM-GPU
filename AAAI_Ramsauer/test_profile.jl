using Distributions, Random, Printf

const φ_star = (sqrt(5)-1)/2
const g_max  = 0.5*log(φ_star) + φ_star
const STD_NORMAL = Normal()

function compute_K(α, N)
    φ_c = α + g_max
    half = (N - 1) / 2
    log_comp = logccdf(Beta(half, half), (1.0 + φ_c) / 2.0)
    log_K = N*α + log_comp
    log_K > 40.0 && return -1
    return max(round(Int, exp(log_K)), 1)
end

function sample_overlaps(α, N, K, rng)
    φ_c = α + g_max
    half = (N - 1) / 2
    tn = truncated(Beta(half, half), (1.0 + φ_c) / 2.0, 1.0)
    z = Vector{Float64}(undef, K)
    for k in 1:K
        z[k] = 2.0 * rand(rng, tn) - 1.0
    end
    sort!(z, rev=true)
    return z
end

@inline function log_Z_hybrid(φ_1, z, α, N)
    omφ² = 1.0 - φ_1*φ_1
    retrieved = N*φ_1
    bulk = N*(α + g_max)
    max_val = max(retrieved, bulk)
    if !isempty(z)
        zmax = φ_1 >= 0 ? z[1] : z[end]
        kept_max = N*(φ_1*zmax + 0.5*(1.0 - zmax*zmax)*omφ²)
        max_val = max(max_val, kept_max)
    end
    s = exp(retrieved - max_val) + exp(bulk - max_val)
    for k in eachindex(z)
        zk = z[k]
        s += exp(N*(φ_1*zk + 0.5*(1.0 - zk*zk)*omφ²) - max_val)
    end
    return max_val + log(s)
end

# F(φ)/N = -(1/N) log Z(φ) - (T/2) log(1-φ²)
function F_per_N(φ, z, α, T, N)
    return -log_Z_hybrid(φ, z, α, N)/N - 0.5*T*log1p(-φ*φ)
end

# LO prediction for comparison
function F_LO(φ, α, T)
    g = α + g_max
    main = max(φ, g)
    return -main - 0.5*T*log1p(-φ*φ)
end

function dump_profile(α, T, N; seed=1)
    rng = MersenneTwister(seed)
    K = compute_K(α, N)
    z = sample_overlaps(α, N, K, rng)
    @printf("\nα=%.3f  T=%.4f  N=%d  K=%d   φ_c=%.4f\n", α, T, N, K, α+g_max)
    @printf("%6s  %12s  %12s  %12s\n", "φ", "F_hyb/N", "F_LO/N", "diff")
    for φ in 0.0:0.05:0.99
        Fh = F_per_N(φ, z, α, T, N)
        Fl = F_LO(φ, α, T)
        @printf("%6.3f  %12.6f  %12.6f  %12.6f\n", φ, Fh, Fl, Fh-Fl)
    end
end

dump_profile(0.50, 0.0625, 100)
dump_profile(0.62, 0.0625, 100)
