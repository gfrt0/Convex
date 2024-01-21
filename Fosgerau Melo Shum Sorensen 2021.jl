#--------------------------------------------------------------------------
# ψ_sim.m: Calculate ψ(p) from  Fosgerau, Melo, Shum, Sørensen (2021),
#            Equation (16), p. 4, up to simulation error.
#--------------------------------------------------------------------------
# AUTHOR: The MATLAB code was written by Jesper Riis-Vestergaard Sørensen, 
# Department of Economics, University of Copenhagen, Denmark.
#   e:  jrvs@econ.ku.dk
#   w:  https://sites.google.com/site/jesperrvs
#
# DATE: 26 May 2022
#
# REFERENCE: When using this software, please cite
#   Fosgerau, Melo, Shum, Sørensen (2021) "Some remarks on CCP-based
#   estimators of dynamic models"
#   Economics Letters, Vol. 204, July 2021.
#   https://doi.org/10.1016/j.econlet.2021.109911

using Distributions, Optim, ForwardDiff, BenchmarkTools

# EXAMPLES: For each of the examples below let
J = 5;                                # number of alternatives (including outside good, if any)
p = collect(1:J) / sum(1:J);          # choice probabilities
S = 1000000;                          # number of simulation draws
γ = Base.MathConstants.eulergamma
𝓋 = zeros(J)
ε = rand(Gumbel(), J, S);             # J x S indep. Gumbel(0,1) draws

function Wₛfast(𝓋, ε)
    r = 0.0 
    @inbounds for j ∈ axes(ε, 2)
        κ = -Inf 
        @inbounds for k ∈ axes(ε, 1) 
            @views κ = κ < 𝓋[k] + ε[k, j] ? 𝓋[k] + ε[k, j] : κ
        end 
        r += κ  
    end 
    return r / size(ε, 2) 
end 

@btime Wₛfast(𝓋, ε)

fₛ(𝒲, 𝓋, p, ε) = exp(𝒲(𝓋, ε)) - 𝓋'p;                # criterion (negative to turn maximization into minimization)

function ψ_simAD(p, ε; opts = Optim.Options(show_trace = false, g_tol = 1e-14, time_limit = 100))
    ψ₀   = zeros(size(p, 1));
    ∇∇fₛ  = TwiceDifferentiable(𝓋 -> fₛ(Wₛfast, 𝓋, p, ε), ψ₀);
    ψ    = optimize(∇∇fₛ, ψ₀, NewtonTrustRegion(), opts);
    return ψ
end

# Burn In Example 
    N = rand(Normal(0, 1), J, 1000)
    ψAD     = ψ_simAD(p, N)
    Optim.minimizer( ψAD )'

# Independent Type 1 Extreme Value
    ε = rand(Gumbel(), J, S);             # J x S indep. Gumbel(0,1) draws
    ψAD     = ψ_simAD(p, ε)
    Optim.minimizer( ψAD )'
    (log.(p) .- γ)'                       # do not forget the euler gamma!

# Independent N(0,1)
    ε = randn(J, S);                      # J x S indep. N(0,1) draws
    ψAD     = ψ_simAD(p, ε)
    Optim.minimizer( ψAD )'

# Correlated Normal
    ρ = .5;
    Σ = zeros(J, J);
    for j=1:J
        for k=1:J       
            Σ[j, k] = ρ^(abs(j-k));       # Toeplitz covariance matrix
        end
    end
    ε = rand(MvNormal(zeros(J), Σ), S);   # N(0_J, Σ) draws
    ψAD     = ψ_simAD(p, ε)
    Optim.minimizer( ψAD )'

# for reference: 
# function ∇Wₛ(𝓋, ε, J)                   # ∇ approximate surplus
#     maxima = map(argmax, eachcol(𝓋 .+ ε));          
#     r = []
#     for i ∈ 1:J
#         r[i] = mean(maxima .== i)
#     end 
#     return r
# end 

# function ∇fₛ!(G, 𝓋, p, ε, J) 
#     G .= ∇Wₛ(𝓋, ε, J) * exp(Wₛ(𝓋, ε)) .- p
# end 
# ∇fₛ!(G, 𝓋) = ∇fₛ!(G, 𝓋, p, ε, J)