"""
Main training script for flow-based generative model.

This script trains the model and generates visualizations using modular plotting and serialization functions.
"""

# ============================================================================
# Setup and Imports
# ============================================================================
using Pkg
Pkg.activate("/Users/hedwignordlinder/Documents/Code/Julia/Karolinska/zGMVisualisation/")
using Revise
using ForwardBackward, Flowfusion, Flux, RandomFeatureMaps, Optimisers, Plots, Distributions, Statistics, Zygote

# Load local modules
include("sexyanimation.jl")
include("serialisation.jl")
include("plotting.jl")

all_plots = []

# ============================================================================
# Model Definition
# ============================================================================

# Provide copy for SwitchingState so gen can broadcast copy.(X0_plot) when tracking
Base.copy(s::SwitchingState) = SwitchingState(copy(s.continuous_state), copy(s.discrete_state))

struct FModel{A}
    layers::A
end
Flux.@layer FModel

function FModel(; embeddim = 128, spacedim = 2, layers = 3)
    # Split the embedding dimension in half for two scales
    half_dim = embeddim ÷ 2
    
    # Scale 1: Low frequency (Global structure)
    rff_low = RandomFourierFeatures(1 => 2*half_dim, 1f0)
    
    # Scale 2: High frequency (Sharp transitions)
    rff_high = RandomFourierFeatures(1 => 2*half_dim, 5f0)
    
    # Combined embedding: Concatenate the outputs, then mix them
    embed_time = Chain(
        Parallel(vcat, rff_low, rff_high),
        Dense(4*half_dim => embeddim, swish)
    )

    embed_state = Chain(RandomFourierFeatures(1 => 4*embeddim, 0.5f0), Dense(4*embeddim => embeddim, swish))
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
    
    # Reshape t to (1, N) for broadcasting/embedding
    # t can be a scalar or a vector matching batch size
    tv = t isa AbstractArray ? reshape(t, 1, :) : fill(t, 1, size(tXt, 2))
    
    x = l.embed_time(tv) .+ l.embed_state(tXt)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    tXt .+ l.decode(x) .* (1.05f0 .- tv) 
end

# ============================================================================
# Training Configuration
# ============================================================================

model = FModel(embeddim = 1024, layers = 4, spacedim = 1)
T = Float32

# Distributions
X0_continuous_distribution = Normal(0,1)
X1_continuous_distribution = MixtureModel([Normal(-0.5,0.1), Normal(0.6,0.5), Normal(0.9,0.1)], [0.2, 0.4, 0.4])

# Sampling functions
sampleX0(n_samples) = SwitchingState(
    ContinuousState(T.(rand(X0_continuous_distribution, 1, n_samples))), 
    DiscreteState(2, rand(1:2, 1, n_samples))
)
sampleX1(n_samples) = SwitchingState(
    ContinuousState(T.(rand(X1_continuous_distribution, 1, n_samples))), 
    DiscreteState(2, ones(Int, 1, n_samples))
)

# Training parameters
n_samples = 4096
μ = 3f0
rate_divergence_time = 0.67

# Rate function for bridging
# Returns (rate_to_x1, rate_to_neg_x1) - the rates for bridging to each endpoint
function rate_func(t, state::ForwardBackward.SwitchingState)
    x = state.continuous_state.state[1]  
    base_rate = 2 * μ
    rate_multiplier = x < 0 ? 1.0f0 : 3.0f0
    r_to_x1 = base_rate * rate_multiplier * (t < rate_divergence_time ? 1/(rate_divergence_time-t) : Inf)
    r_to_neg_x1 = base_rate * rate_multiplier * max(0, rate_divergence_time-t)
    
    return (r_to_x1, r_to_neg_x1)
end

# Define the process
P = SwitchingProcess(Deterministic(), rate_func)

# ============================================================================
# Training Loop
# ============================================================================

println("Starting training...")

# Optimizer setup
eta_max = 5e-5
eta_min = 1e-9
opt_state = Flux.setup(AdamW(eta = eta_max), model)

iters = 5000
warmup_iters = 200
rampup_iters = 50
losses = Float32[]

for i in 1:iters
    # Rampup -> Constant High -> Linear Decay
    if i <= rampup_iters
         lr = eta_max * (i / rampup_iters)
    elseif i <= warmup_iters
        lr = eta_max
    else
        progress = (i - warmup_iters) / (iters - warmup_iters)
        lr = eta_max - progress * (eta_max - eta_min)
    end
    Optimisers.adjust!(opt_state, lr)
    
    # Sample a batch of data
    X0 = sampleX0(n_samples)
    X1 = sampleX1(n_samples)
    t = rand(T, n_samples)
    # Construct the bridge
    Xt = endpoint_conditioned_sample(X1, X0, P, t; δt = 1e-3)
    # Gradient and update
    l, (∇model,) = Flux.withgradient(model) do m
        ŷ = m(t, Xt.continuous_state)
        x1 = tensor(X1.continuous_state)
        # +1 if latent==1 (bridge to X1), -1 otherwise (bridge to -X1)
        sgn = ifelse.(Xt.discrete_state.state .== 1, one(eltype(x1)), -one(eltype(x1)))
        X1_signed = ContinuousState(expand(x1, ndims(ŷ)) .* expand(sgn, ndims(ŷ)))
        # Upweight by 2x when prediction < 0
        #c = scalefloss(P.continuous_process, t) .* ifelse.(ŷ .< 0, 1.5f0, 1f0)
        
        floss(P.continuous_process, ŷ, X1_signed, scalefloss(P.continuous_process, t))
        #floss(P.continuous_process, ŷ, X1_signed, 1.0f0)
    end
    Flux.update!(opt_state, model, ∇model)
    push!(losses, l)
   
    if i < 100 || i % 100 == 0
        @info "iter=$i loss=$(l) lr=$(lr)"
    end
end

println("Training complete")

# ============================================================================
# Loss Plot
# ============================================================================

begin
    println("\nGenerating loss plot...")
    data_dir = "plot_data"
    mkpath(data_dir)
    save_training_data(losses; data_dir=data_dir)
    
    plt_loss = plot_loss(losses)
    savefig(plt_loss, "loss.png")
    display(plt_loss)
    push!(all_plots, plt_loss)
    println("  ✓ Loss plot")
end

# ============================================================================
# Generate Visualization Data
# ============================================================================

println("\nGenerating visualization data...")

# 1. Marginal generation
begin
    println("  - Marginal generation trajectories...")
    n_plot_samples = 1000
    quantiles = collect(range(0.01, 0.99; length = n_plot_samples))
    x0_vals = quantile.(Ref(X0_continuous_distribution), quantiles)
    X0_plot = ContinuousState(T.(reshape(x0_vals, 1, :)))
    paths = Tracker()
    samples = gen(Deterministic(), X0_plot, model, 0f0:0.005f0:1f0; tracker = paths)

    # Stack tracked trajectories
    tvec = Float64.(stack_tracker(paths, :t))
    xttraj = Float64.(stack_tracker(paths, :xt))
    final_samples = vec(Float64.(tensor(samples)))
    
    # Save and plot within this scope
    println("  - Saving marginal generation data...")
    save_marginal_generation_data(tvec, xttraj, x0_vals, final_samples; data_dir="plot_data")
    
    println("  - Plotting marginal generation...")
    # Reshape 3D to 2D for plotting: (xdim, nsamples, nsteps) -> (nsteps, nsamples)
    xttraj_2d = reshape(xttraj, size(xttraj, 1) * size(xttraj, 2), size(xttraj, 3))'
    plt_marginal = plot_marginal_generation(tvec, xttraj_2d, x0_vals, final_samples; target_dist=X1_continuous_distribution)
    savefig(plt_marginal, "marginal_generation.png")
    display(plt_marginal)
    push!(all_plots, plt_marginal)
    println("  ✓ Marginal generation plot")
end

# 2. Conditional trajectories
begin
    println("  - Conditional trajectories...")
    n_cond_samples = 100
    quantiles_cond = collect(range(0.01, 0.99; length = n_cond_samples))
    x0_vals_cond = quantile.(Ref(X0_continuous_distribution), quantiles_cond)
    X0_cond = SwitchingState(
        ContinuousState(T.(reshape(x0_vals_cond, 1, :))),
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
        end

        Y[i, :] = y_i
        D[i, :] = d_i
        push!(y_final, y_i[end])
        push!(x1_targets, X1_cond.continuous_state.state[1, i])
    end

    # Compute density for conditional plot
    y_min_plot = minimum(y_final) - 0.2
    y_max_plot = maximum(y_final) + 0.2
    y_grid = collect(range(y_min_plot, y_max_plot; length = 200))
    density_vals = pdf.(Ref(X1_continuous_distribution), y_grid)
    
    # Save and plot within this scope
    println("  - Saving conditional trajectories data...")
    save_conditional_trajectories_data(tvec_ref[], Y, D, y_final, x1_targets, y_grid, density_vals; data_dir="plot_data")
    
    println("  - Plotting conditional trajectories...")
    plt_conditional = plot_conditional_trajectories(tvec_ref[], Y, D, y_final, x1_targets, y_grid, density_vals)
    savefig(plt_conditional, "conditional_generation.png")
    display(plt_conditional)
    push!(all_plots, plt_conditional)
    println("  ✓ Conditional trajectories plot")
end

# 3. Final density comparison
begin
    println("  - Final density comparison...")
    n_hist_samples = 5000
    X0_hist = ContinuousState(T.(rand(X0_continuous_distribution, 1, n_hist_samples)))
    samples_hist = gen(Deterministic(), X0_hist, model, 0f0:0.005f0:1f0)
    final_hist = vec(Float64.(tensor(samples_hist)))

    x_min = minimum(final_hist)
    x_max = maximum(final_hist)
    xgrid = collect(range(x_min, x_max; length = 400))
    pdf_vals = pdf.(Ref(X1_continuous_distribution), xgrid)
    
    # Save and plot within this scope
    println("  - Saving final density data...")
    save_final_density_data(final_hist, xgrid, pdf_vals; data_dir="plot_data")
    
    println("  - Plotting final density...")
    plt_density = plot_final_density(final_hist, xgrid, pdf_vals)
    savefig(plt_density, "conditional_density_compare.png")
    display(plt_density)
    push!(all_plots, plt_density)
    println("  ✓ Final density plot")
end

# ============================================================================
# Save Combined PDF
# ============================================================================

println("\nSaving combined PDF...")
save_all_plots(all_plots, "all_plots.pdf")

println("\n" * "="^60)
println("All tasks complete!")
println("  - Model trained for $iters iterations")
println("  - Data saved to '$data_dir/' directory")
println("  - Individual plots saved as PNG files")
println("  - Combined plots saved to 'all_plots.pdf'")
println("="^60)

