using Pkg
Pkg.activate("/Users/hedwignordlinder/Documents/Code/Julia/Karolinska/zGMVisualisation/")
using Revise
using ForwardBackward, Flowfusion, Flux, RandomFeatureMaps, Optimisers, Plots, Distributions

struct FModel{A}
    layers::A
end
Flux.@layer FModel
function FModel(; embeddim = 128, spacedim = 2, layers = 3)
    embed_time = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode = Dense(embeddim => spacedim)
    layers = (; embed_time, embed_state, ffs, decode)
    FModel(layers)
end
function (f::FModel)(t, Xt)
    l = f.layers
    # Allow Xt to be a ContinuousState or a raw array/vector. Ensure 2D (1×N).
    tXt_raw = Xt isa ForwardBackward.ContinuousState ? Xt.state : tensor(Xt)
    tXt = ndims(tXt_raw) == 1 ? reshape(tXt_raw, 1, length(tXt_raw)) : tXt_raw
    tv = zero(tXt[1:1,:]) .+ expand(t, ndims(tXt))
    x = l.embed_time(tv) .+ l.embed_state(tXt)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    tXt .+ l.decode(x) .* (1.05f0 .- expand(t, ndims(tXt))) 
end

model = FModel(embeddim = 128, layers = 3, spacedim = 1)

T = Float32
X0_continuous_distribution = Normal(0,1)
X1_continuous_distribution = MixtureModel([Normal(1,1), Normal(0,0.5), Normal(5,0.1)], [0.2,0.7,0.1])

sampleX0(n_samples) = SwitchingState(ContinuousState(T.(rand(X0_continuous_distribution, 1, n_samples))), DiscreteState(2, rand(1:2, 1, n_samples))) # In the beginning, we randomly either bridge to the endpoint (1) or to its negation (2)
sampleX1(n_samples) = SwitchingState(ContinuousState(T.(rand(X1_continuous_distribution, 1, n_samples))), DiscreteState(2, ones(Int, 1, n_samples))) # Always end bridging towards the endpoint (1)

n_samples = 400 # Who knows what will be computationally tractable? 

P = SwitchingProcess(BrownianMotion(0.15f0), UniformDiscrete(0.5f0))

# Optimiser 
eta = 1e-3
opt_state = Flux.setup(AdamW(eta = eta), model)

iters = 1000
for i in 1:iters
    # Sample a batch of data
    X0 = sampleX0(n_samples)
    X1 = sampleX1(n_samples)
    t = rand(T, n_samples)
    # Construct the bridge
    Xt = endpoint_conditioned_sample(X1, X0, P, t)
    # Gradient and update
    l, (∇model,) = Flux.withgradient(model) do m
        ŷ = m(t, Xt.continuous_state)
        x1 = tensor(X1.continuous_state)
        # +1 if latent==1 (bridge to X1), -1 otherwise (bridge to -X1)
        sgn = ifelse.(Xt.discrete_state.state .== 1, one(eltype(x1)), -one(eltype(x1)))
        X1_signed = ContinuousState(expand(x1, ndims(ŷ)) .* expand(sgn, ndims(ŷ)))
        floss(P.continuous_process, ŷ, X1_signed, scalefloss(P.continuous_process, t))
    end
    Flux.update!(opt_state, model, ∇model)
    if i % 100 == 0
        @info "iter=$i loss=$(l)"
    end
end

n_inference_samples = 20000
X0 = ContinuousState(T.(rand(X0_continuous_distribution, 1, n_inference_samples)))

paths = Tracker()


samples = gen(BrownianMotion(0.0f0, 1.0f0), X0, model, 0f0:0.005f0:1f0, tracker = paths)
tvec = stack_tracker(paths, :t)
xttraj = stack_tracker(paths, :xt)

# Trajectories from t ∈ [0,1] and a right-anchored histogram at t=1
vals = vec(samples.state)
pl = plot(; legend = :topright)
for i in 1:10:n_inference_samples
    plot!(tvec, xttraj[1,i,:], color = "red", label = i==1 ? "Trajectory" : :none, alpha = 0.05)
end

# Build a simple pdf-normalized histogram without extra deps
nbins = 40
vmin = minimum(vals)
vmax = maximum(vals)
edges = collect(range(vmin, vmax; length = nbins + 1))
binwidth = edges[2] - edges[1]
densities = zeros(Float64, nbins)
for i in 1:nbins
    lo = edges[i]
    hi = edges[i+1]
    densities[i] = count(x -> (x >= lo) && (x < hi), vals)
end
densities ./= (length(vals) * binwidth)  # normalize to pdf

# Draw the histogram as horizontal bars anchored at t = 1
α = 0.2
for i in 1:nbins
    xcoords = [1, 1 + α * densities[i], 1 + α * densities[i], 1]
    ycoords = [edges[i], edges[i], edges[i+1], edges[i+1]]
    plot!(xcoords, ycoords; seriestype = :shape, c = :gray, a = 0.35, lc = :gray, label = i == 1 ? "t=1 hist" : :none)
end

xlabel!("t")
ylabel!("x")
xlims!(0, 1 + α * (isempty(densities) ? 0 : maximum(densities)) * 1.2)
title!("Marginal trjaectories")
display(pl)