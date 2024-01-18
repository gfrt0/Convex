using Distributions, LinearAlgebra, Optim, FastGaussQuadrature

J = 5            # number of products 
M = 4            # number of product characteristics 
N = 10000        # number of consumers 

MvN = MvNormal(zeros(M), Matrix{Float64}(I, M, M))
γ = Base.MathConstants.eulergamma

## Random Coefficient Mixed Logit

β = rand(Uniform(), M)
Z = rand(MvN, J) 
ν = rand(MvN, N)

# computing expected surplus and market shares
𝓊(x, ϵ)   = x .+ ϵ
Σp1(x, ϵ) = 1.0 .+ sum( exp.( 𝓊(x, ϵ) ), dims = 1)
𝒰(x, ϵ)   = γ + mean( log.( Σp1(x, ϵ) ) )
ς(x, ϵ)   = mean( exp.( 𝓊(x, ϵ) ) ./ Σp1(x, ϵ), dims = 2) # do not sum to 1 because outside good is not included.

f(x, ϵ, p) = 𝒰(x, ϵ) .- x'vec(p)

function opt(p, ϵ; opts = Optim.Options(show_trace = true, g_tol = 1e-14, time_limit = 100))
    ψ = optimize(x -> f(x, ϵ, p), zeros(size(p, 1)), NelderMead(), opts); 
end

x = Z'β
ϵ = Z'ν
σ = ς(x, ϵ)
xhat = opt(σ, ϵ)

[Optim.minimizer(xhat) x Optim.minimizer(xhat) .- x]'

## Pure Characteristics Model 

β = [1.0, rand(Uniform(), M - 1)...]
Z = rand(MvN, J) 
ν = rand(MvNormal(zeros(M - 1), Matrix{Float64}(I, M - 1, M - 1)), N)

GH = 50 
nodes, weights = gausshermite( GH )
nodes = sqrt(2.0) .* nodes
weights = weights ./ sqrt(π)

# computing expected surplus 
𝓋(x, Z, β, ν, μ)  = x .+ Z[1:1, :]'μ .+ Z[2:M, :]'ν
𝓊(x, Z, β, ν, μ)  = map(maximum, eachcol(𝓋(x, Z, β, ν, μ)))
𝓊⁺(x, Z, β, ν, μ) = max.( 0, 𝓊(x, Z, β, ν, μ) )
∫𝓊⁺(x, Z, β, ν)   = sum( permutedims(hcat([𝓊⁺(x, Z, β, ν, nodes[i]) .* weights[i] for i ∈ 1:GH]...)), dims = 1 )
𝒰(x, Z, β, ν)     = mean( ∫𝓊⁺(x, Z, β, ν) )

# computing market shares 
I𝓊(x, Z, β, ν, μ)  = 𝓋(x, Z, β, ν, μ) .>= ones(size(Z, 2)) * 𝓊⁺(x, Z, β, ν, μ)' 
σⱼ(x, Z, β, ν)  = mean( sum([I𝓊(x, Z, β, ν, nodes[i]) .* weights[i]  for i ∈ 1:GH]), dims = 2 )

f(x, Z, β, ν, p) = 𝒰(x, Z, β, ν) .- x'vec(p)

function opt(Z, β, ν, p; opts = Optim.Options(show_trace = true, g_tol = 1e-14, time_limit = 100))
    ψ = optimize(x -> f(x, Z, β, ν, p), zeros(size(p, 1)), NelderMead(), opts); 
end

x = Z'β
p = σⱼ(x, Z, β, ν)

xhat = opt(Z, β, ν, p)

[Optim.minimizer(xhat) x Optim.minimizer(xhat) .- x]'