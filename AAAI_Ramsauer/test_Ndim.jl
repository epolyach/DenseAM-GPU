using Distributions, LinearAlgebra, Random, Printf

const φ_star = (sqrt(5)-1)/2
const g_max  = 0.5*log(φ_star) + φ_star
phi_eq(T) = 0.5 * (-T + sqrt(T^2 + 4))

function compute_K(α, N)
    φ_c = α + g_max
    half = (N-1)/2
    log_comp = logccdf(Beta(half, half), (1.0 + φ_c)/2.0)
    log_K = N*α + log_comp
    log_K > 40.0 && return -1
    return max(round(Int, exp(log_K)), 1)
end

function build_patterns(α, N, K, rng)
    φ_c = α + g_max
    half = (N-1)/2
    tn = truncated(Beta(half, half), (1.0 + φ_c)/2.0, 1.0)
    sN = sqrt(N)
    P = Matrix{Float64}(undef, K, N)
    u = Vector{Float64}(undef, N-1)
    for k in 1:K
        φ_1μ = 2.0 * rand(rng, tn) - 1.0
        P[k, 1] = sN * φ_1μ
        randn!(rng, u)
        unrm = norm(u)
        scale = sN * sqrt(max(0.0, 1.0 - φ_1μ^2)) / unrm
        for i in 2:N
            P[k, i] = scale * u[i-1]
        end
    end
    return P
end

@inline function log_Z(dots, retr, log_bulk)
    max_val = max(retr, log_bulk, maximum(dots))
    s = exp(retr - max_val) + exp(log_bulk - max_val)
    for v in dots
        s += exp(v - max_val)
    end
    return max_val + log(s)
end

function run_mc(α, T, N, P, log_bulk, n_eq, n_samp, rng)
    K = size(P, 1)
    sN = sqrt(N)
    a = min(phi_eq(T), 0.999)
    x = zeros(N); x[1] = sN * a
    u = randn(rng, N-1); u ./= norm(u)
    scale = sN * sqrt(max(0.0, 1.0 - a^2))
    for i in 2:N
        x[i] = scale * u[i-1]
    end
    dots = P * x
    retr = sN * x[1]
    logZ = log_Z(dots, retr, log_bulk)
    σ = max(0.01, 2.0 * sqrt(T/N))
    η = Vector{Float64}(undef, N)
    x_new = Vector{Float64}(undef, N)
    dots_new = Vector{Float64}(undef, K)
    sum_φ = 0.0; sum_φ2 = 0.0; n_acc = 0
    for step in 1:(n_eq + n_samp)
        randn!(rng, η)
        for i in 1:N
            x_new[i] = x[i] + σ * η[i]
        end
        nrm = norm(x_new)
        for i in 1:N
            x_new[i] *= sN / nrm
        end
        mul!(dots_new, P, x_new)
        retr_new = sN * x_new[1]
        logZ_new = log_Z(dots_new, retr_new, log_bulk)
        log_acc = (logZ_new - logZ) / T
        if log_acc >= 0.0 || log(rand(rng)) < log_acc
            copyto!(x, x_new); copyto!(dots, dots_new)
            retr = retr_new; logZ = logZ_new
            n_acc += 1
        end
        if step > n_eq
            φ_1 = x[1] / sN
            sum_φ  += φ_1
            sum_φ2 += φ_1 * φ_1
        end
    end
    mφ = sum_φ / n_samp
    return mφ, sum_φ2/n_samp - mφ^2, n_acc / (n_eq + n_samp)
end

function bench(α, T, N, n_eq, n_samp)
    rng_p = MersenneTwister(42)
    K = compute_K(α, N)
    P = build_patterns(α, N, K, rng_p)
    log_bulk = N*α + N*g_max
    rng_m = MersenneTwister(7)
    t0 = time()
    mφ, vφ, acc = run_mc(α, T, N, P, log_bulk, n_eq, n_samp, rng_m)
    dt = time() - t0
    @printf("α=%.2f  T=%.4f  N=%d  K=%d  ⟨φ⟩=%.4f  var=%.5f  acc=%.3f  φ_eq=%.4f  time=%.2fs\n",
            α, T, N, K, mφ, vφ, acc, phi_eq(T), dt)
end

@info "warmup"
bench(0.50, 0.05, 100, 200, 500)
@info "real bench"
bench(0.50, 0.05, 100, 4000, 16000)
bench(0.55, 0.04, 100, 4000, 16000)
bench(0.62, 0.005, 100, 4000, 16000)
