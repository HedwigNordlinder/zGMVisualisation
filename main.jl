using Pkg
Pkg.activate("/Users/hedwignordlinder/Documents/Code/Julia/Karolinska/zGMVisualisation/")
using Revise
using ForwardBackward, Flowfusion, Flux, RandomFeatureMaps, Optimisers, Plots, Distributions, Statistics
include("sexyanimation.jl")

# Provide copy for SwitchingState so gen can broadcast copy.(X0_plot) when tracking
Base.copy(s::SwitchingState) = SwitchingState(copy(s.continuous_state), copy(s.discrete_state))

struct FModel{A}
    layers::A
end
Flux.@layer FModel
function FModel(; embeddim = 128, spacedim = 2, layers = 3)
    embed_time = Chain(RandomFourierFeatures(1 => 4*embeddim, 1f0), Dense(4*embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(1 => 4*embeddim, 1f0), Dense(4*embeddim => embeddim, swish))
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

model = FModel(embeddim = 256, layers = 4, spacedim = 1)

T = Float32
X0_continuous_distribution = Uniform(0,4)
X1_continuous_distribution = MixtureModel([Normal(1,1), Normal(5,0.5),Normal(-2,0.2)], [0.5,0.4,0.1])

sampleX0(n_samples) = SwitchingState(ContinuousState(T.(rand(X0_continuous_distribution, 1, n_samples))), DiscreteState(2, rand(1:2, 1, n_samples))) # In the beginning, we randomly either bridge to the endpoint (1) or to its negation (2)
sampleX1(n_samples) = SwitchingState(ContinuousState(T.(rand(X1_continuous_distribution, 1, n_samples))), DiscreteState(2, ones(Int, 1, n_samples))) # Always end bridging towards the endpoint (1)

n_samples = 1600 # Who knows what will be computationally tractable? 

# Rate function for bridging
# Returns (rate_to_x1, rate_to_neg_x1) - the rates for bridging to each endpoint
# Base rate μ = 0.5, higher when continuous state is above 0
μ = 0.5f0

# Rate function takes (t, state) and returns (rate_to_x1, rate_to_neg_x1)
function rate_func(t, state::ForwardBackward.SwitchingState)
    x = state.continuous_state.state[1]  
    base_rate =2* μ
    rate_multiplier = x < 0 ? 1.0f0 : 5.0f0
    
    r_to_x1 = base_rate * rate_multiplier * (t < 0.8 ? 1/(0.8-t) : Inf)
    r_to_neg_x1 = base_rate * rate_multiplier * max(0,0.8-t)
    
    return (r_to_x1, r_to_neg_x1)
end

P = SwitchingProcess(Deterministic(), rate_func)

# Optimiser 
eta = 1e-3
opt_state = Flux.setup(AdamW(eta = eta), model)

iters = 4000
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
    
    Optimisers.adjust!(opt_state, eta - 1e-3/iters)
    if i % 100 == 0
        @info "iter=$i loss=$(l)"
    end
end

# ---------------------------------------------------------------------
# Marginal generation plot (100 samples) - run after training
# ---------------------------------------------------------------------
begin
    n_plot_samples = 100
    quantiles = collect(range(0.01, 0.99; length = n_plot_samples))
    x0_vals = quantile.(Ref(X0_continuous_distribution), quantiles)
    X0_plot = ContinuousState(T.(reshape(x0_vals, 1, :)))  # deterministic process: only continuous part matters
    paths = Tracker()
    samples = gen(Deterministic(), X0_plot, model, 0f0:0.005f0:1f0; tracker = paths)

    # Stack tracked trajectories (dims: xdim × nsamples × nsteps)
    tvec = Float64.(stack_tracker(paths, :t))
    xttraj = Float64.(stack_tracker(paths, :xt))
    xdim, nsamples, nsteps = size(xttraj)

    # Append the final generated state at t = 1 for nicer alignment with histogram
    final_state = Float64.(tensor(samples))  # xdim × nsamples
    t_full = [tvec; 1.0]
    xt_full = cat(xttraj, reshape(final_state, xdim, nsamples, 1); dims = 3)

    # Extract 1D trajectories for plotting
    Y = Array{Float64}(undef, nsamples, size(xt_full, 3))
    for i in 1:nsamples
        Y[i, :] = vec(xt_full[1, i, :])
    end

    # Histogram of final-time values
    final_vals = vec(Y[:, end])
    nbins = 35
    edges = collect(range(minimum(final_vals), maximum(final_vals); length = nbins + 1))
    binwidth = edges[2] - edges[1]
    densities = zeros(Float64, nbins)
    for i in 1:nbins
        lo, hi = edges[i], edges[i + 1]
        densities[i] = count(x -> (x >= lo) && (x < hi), final_vals)
    end
    if !isempty(densities)
        densities ./= (length(final_vals) * binwidth)  # approximate pdf height
    end

    α = 0.25
    x_right = 1 + α * (isempty(densities) ? 0 : maximum(densities)) * 1.25
    y_min = minimum(Y)
    y_max = maximum(Y)

    plt = plot(; legend = :none, size = (900, 600), background_color = :white)
    for i in 1:nsamples
        plot!(plt, t_full, Y[i, :]; color = :royalblue, alpha = 0.2, linewidth = 1)
    end

    # Right-anchored histogram at t = 1
    for i in 1:nbins
        xcoords = [1, 1 + α * densities[i], 1 + α * densities[i], 1]
        ycoords = [edges[i], edges[i], edges[i + 1], edges[i + 1]]
        plot!(plt, xcoords, ycoords; seriestype = :shape, c = :gray, a = 0.45, lc = :gray)
    end

    xlabel!(plt, "t")
    ylabel!(plt, "x")
    xlims!(plt, 0, x_right)
    ylims!(plt, y_min, y_max)
    title!(plt, "Marginal trajectories")
    savefig(plt, "marginal_generation.png")
    display(plt)
end

# ---------------------------------------------------------------------
# Conditional generation plot (endpoint-conditioned samples)
# ---------------------------------------------------------------------
begin
    n_cond_samples = 100
    quantiles = collect(range(0.01, 0.99; length = n_cond_samples))
    x0_vals = quantile.(Ref(X0_continuous_distribution), quantiles)
    X0_cond = SwitchingState(
        ContinuousState(T.(reshape(x0_vals, 1, :))),
        DiscreteState(2, rand(1:2, 1, n_cond_samples)),
    )
    X1_cond = sampleX1(n_cond_samples)

    δt = 5f-3
    local tvec_ref = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    local Y = nothing
    local D = nothing
    local y_final = Float64[]
    local x1_targets = Float64[]

    for i in 1:n_cond_samples
        t_track = Float64[]
        y_track = Float64[]
        d_track = Int[]

        track = function (t, xt, _)
            push!(t_track, Float64(t))
            push!(y_track, Float64(xt[1].state[1]))
            push!(d_track, Int(xt[2].state[1]))
            return nothing
        end

        X0_i = SwitchingState(
            ContinuousState(view(X0_cond.continuous_state.state, 1:1, i:i)),
            DiscreteState(X0_cond.discrete_state.K, view(X0_cond.discrete_state.state, 1:1, i:i)),
        )
        X1_i = SwitchingState(
            ContinuousState(view(X1_cond.continuous_state.state, 1:1, i:i)),
            DiscreteState(X1_cond.discrete_state.K, view(X1_cond.discrete_state.state, 1:1, i:i)),
        )

        _ = endpoint_conditioned_sample(X1_i, X0_i, P, T(1); δt = δt, tracker = track)
        tvec_i = t_track
        y_i = y_track
        d_i = d_track

        if tvec_ref[] === nothing
            tvec_ref[] = tvec_i
            Y = Array{Float64}(undef, n_cond_samples, length(tvec_ref[]))
            D = Array{Int}(undef, n_cond_samples, length(tvec_ref[]))
        else
            @assert length(tvec_i) == length(tvec_ref[]) "Conditional trajectories must share the same time grid."
        end

        Y[i, :] = y_i
        D[i, :] = d_i
        push!(y_final, y_i[end])
        push!(x1_targets, X1_cond.continuous_state.state[1, i])
    end
    @assert tvec_ref[] !== nothing

    nbins = 35
    edges = collect(range(minimum(y_final), maximum(y_final); length = nbins + 1))
    binwidth = edges[2] - edges[1]
    densities = zeros(Float64, nbins)
    for i in 1:nbins
        lo = edges[i]
        hi = edges[i + 1]
        densities[i] = count(x -> (x >= lo) && (x < hi), y_final)
    end
    if !isempty(densities)
        densities ./= (length(y_final) * binwidth)
    end

    α = 0.25
    x_right = 1 + α * (isempty(densities) ? 0 : maximum(densities)) * 1.25
    y_min = min(minimum(Y), minimum(x1_targets))
    y_max = max(maximum(Y), maximum(x1_targets))

    plt = plot(; legend = :none, size = (900, 600), background_color = :white)
    state_colors = Dict(1 => :seagreen, 2 => :red)
    for i in 1:n_cond_samples
        for j in 1:(length(tvec_ref[]) - 1)
            seg_color = get(state_colors, D[i, j], :black)
            plot!(plt, tvec_ref[][j:j+1], Y[i, j:j+1]; color = seg_color, alpha = 0.25, linewidth = 1.2)
        end
    end
    scatter!(plt, fill(1.0, length(x1_targets)), x1_targets;
             color = :orange, markerstrokecolor = :black, markersize = 4, alpha = 0.8)

    for i in 1:nbins
        xcoords = [1, 1 + α * densities[i], 1 + α * densities[i], 1]
        ycoords = [edges[i], edges[i], edges[i + 1], edges[i + 1]]
        plot!(plt, xcoords, ycoords; seriestype = :shape, c = :gray, a = 0.45, lc = :gray)
    end

    xlabel!(plt, "t")
    ylabel!(plt, "x")
    xlims!(plt, 0, x_right)
    ylims!(plt, y_min, y_max)
    title!(plt, "Conditional trajectories (endpoint-conditioned)")
    savefig(plt, "conditional_generation.png")
    display(plt)
end
