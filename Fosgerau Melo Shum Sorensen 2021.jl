#--------------------------------------------------------------------------
# ψ_sim.m: Calculate ψ(p) from  Fosgerau, Melo, Shum, Sørensen (2021),
#            Equation (16), p. 4, up to simulation error.
#--------------------------------------------------------------------------
#
# DESCRIPTION: Calculate a solution ψ_S(p) to the problem
#           minimize{exp(W_S(v)) - v'*p subject to v in R^J}
# where W_S is the empirical approximation 
#           W_S(v):=(1/S)sum_{s=S}^{S} max_j{v_j+ε_{sj}}
# to the surplus W(v):=E[max_j{v_j+ε_j}]
# based on S independent draws from the joint distribution of
# ε=(ε_1,...,ε_J).
#
# INPUT ARGUMENTS:
# p:        J x 1 vector of choice probabilities (p_j>0 all j)
# ε:      J x S matrix of simulation draws
#
# OUTPUT ARGUMENT: 
# ψ: The scalar ψ_S(p)
#
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

Wₛ(𝓋, ε) = mean( map(maximum, eachcol(𝓋 .+ ε)) );    # approximate surplus
fₛ(𝓋, p, ε) = exp(Wₛ(𝓋, ε)) .- 𝓋'*p;                 # criterion (negative to turn maximization into minimization)

function ψ_sim(p, ε; opts = Optim.Options(show_trace = false, g_tol = 1e-14))
    ψ₀   = zeros(size(p, 1));                                        # starting value
    ψ    = optimize(𝓋 -> fₛ(𝓋, p, ε), ψ₀, NelderMead(), opts);       # optimize
    return ψ
end

function ψ_simAD(p, ε; opts = Optim.Options(show_trace = false, g_tol = 1e-14))
    ψ₀   = zeros(size(p, 1));                                       # starting value
    ∇∇fₛ  = TwiceDifferentiable(𝓋 -> fₛ(𝓋, p, ε), ψ₀)
    ψ    = optimize(∇∇fₛ, ψ₀, NewtonTrustRegion(), opts);            # optimize
    return ψ
end

N = rand(Normal(0, 1), J, 1000)

Optim.minimizer( ψ_sim(p, N) )
Optim.minimizer( ψ_simAD(p, N) )

# Independent Type 1 Extreme Value
    ε = rand(Gumbel(), J, S);             # J x S indep. Gumbel(0,1) draws
    ψ = ψ_simAD(p, ε);
    Optim.minimizer(ψ)
    log.(p) .- γ                          # do not forget the euler gamma!

# Independent N(0,1)
    ε = randn(J, S);                      # J x S indep. N(0,1) draws
    ψ = ψ_simAD(p, ε);
    Optim.minimizer(ψ)

# Correlated Normal
    ρ = .5;
    Σ = zeros(J, J);
    for j=1:J
        for k=1:J       
            Σ[j, k] = ρ^(abs(j-k));       # Toeplitz covariance matrix
        end
    end
    ε = rand(MvNormal(zeros(J), Σ), S);   # N(0_J, Σ) draws
    ψ = ψ_simAD(p, ε);
    Optim.minimizer(ψ)

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