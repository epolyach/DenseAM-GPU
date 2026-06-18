#=
Numerical validation of the closed-form identity

  S_spur / M = ∫_{-1}^{1} h(φ) exp[β N (φ-1)] dφ
             = Γ(N/2) (2/(β N))^{N/2-1} e^{-β N} I_{N/2-1}(β N),

with h(φ) = C_N (1-φ²)^{(N-3)/2}, C_N = Γ(N/2)/[√π Γ((N-1)/2)].

β = 1 throughout.
=#

using SpecialFunctions
using QuadGK
using Printf

const β = 1.0

# log h(φ) = log C_N + (N-3)/2 · log(1-φ²)
log_CN(N) = loggamma(N/2) - 0.5*log(π) - loggamma((N-1)/2)

# Numerical: quadrature on the original integral
function numerical_integral(N)
    lc = log_CN(N)
    f(φ) = exp(lc + ((N-3)/2)*log(1 - φ^2) + β*N*(φ - 1))
    val, _ = quadgk(f, -1.0, 1.0; rtol = 1e-14, atol = 0.0)
    return val
end

# Closed form using besselix(ν, z) = exp(-z) I_ν(z) to avoid overflow
function closedform(N)
    ν = N/2 - 1
    return gamma(N/2) * (2/(β*N))^ν * besselix(ν, β*N)
end

# Saddle (Laplace) prediction: exp[N(g_max(β) - β)]
const φ_star = (sqrt(5) - 1)/2
const g_max  = 0.5*log(1 - φ_star^2) + φ_star   # at β = 1

saddle(N) = exp(N*(g_max - β))

println("β = 1, exact and Laplace comparison")
println()
@printf("%5s  %24s  %24s  %12s  %24s\n",
        "N", "quadrature", "Bessel closed form", "rel. diff", "Laplace e^{N(g_max-β)}")
println(repeat("-", 100))
for N in (7, 20, 100)
    a = numerical_integral(N)
    b = closedform(N)
    s = saddle(N)
    @printf("%5d  %.18e  %.18e  %.2e  %.18e\n",
            N, a, b, abs(a-b)/abs(b), s)
end
