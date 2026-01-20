using Pkg
Pkg.activate("/Users/hedwignordlinder/Documents/Code/Julia/Karolinska/zGMVisualisation/")
using Revise
using ForwardBackward, Flowfusion, Flux, RandomFeatureMaps, Optimisers, Plots, Distributions, Statistics, Zygote
include("sexyanimation.jl")
all_plots = []

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
    # We define a custom layer or closure for this logic
    embed_time = Chain(
        Parallel(vcat, rff_low, rff_high),
        Dense(4*half_dim => embeddim, swish) # 2 * (2*half) = 4*half = 2*embeddim input? No, check dims carefully.
        # RFF(1=>M) outputs M features. 
        # Here: low outputs 2*half, high outputs 2*half. Total = 2*embeddim.
        # Dense should map 2*embeddim => embeddim.
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
    # tv = Zygote.@ignore similar(tXt_raw, 1, size(tXt, 2)) .= t 
    # Similar is a trick that puts it on le gpu
    
  
    
    x = l.embed_time(tv) .+ l.embed_state(tXt)
    for ff in l.ffs
        x = x .+ ff(x)
    end
    tXt .+ l.decode(x) .* (1.05f0 .- tv) 
end

model = FModel(embeddim = 1024, layers = 4, spacedim = 1)
T = Float32
X0_continuous_distribution = Normal(0,1)
X1_continuous_distribution =MixtureModel([Normal(-0.5,0.1), Normal(0.6,0.5), Normal(0.9,0.1)], [0.2, 0.4, 0.4])

sampleX0(n_samples) = SwitchingState(ContinuousState(T.(rand(X0_continuous_distribution, 1, n_samples))), DiscreteState(2, rand(1:2, 1, n_samples))) # In the beginning, we randomly either bridge to the endpoint (1) or to its negation (2)
sampleX1(n_samples) = SwitchingState(ContinuousState(T.(rand(X1_continuous_distribution, 1, n_samples))), DiscreteState(2, ones(Int, 1, n_samples))) # Always end bridging towards the endpoint (1)

n_samples = 4096 # Who knows what will be computationally tractable? 

# Rate function for bridging
# Returns (rate_to_x1, rate_to_neg_x1) - the rates for bridging to each endpoint
# Base rate μ = 0.5, higher when continuous state is above 0
μ = 3f0
rate_divergence_time = 0.67
# Rate function takes (t, state) and returns (rate_to_x1, rate_to_neg_x1)
function rate_func(t, state::ForwardBackward.SwitchingState)
    x = state.continuous_state.state[1]  
    base_rate =2* μ
    rate_multiplier = x < 0 ? 1.0f0 : 3.0f0
    #rate_multiplier = 1.0f0
    r_to_x1 = base_rate * rate_multiplier * (t < rate_divergence_time ? 1/(rate_divergence_time-t) : Inf)
    r_to_neg_x1 = base_rate * rate_multiplier * max(0,rate_divergence_time-t)
    
    return (r_to_x1, r_to_neg_x1)
end


P = SwitchingProcess(Deterministic(), rate_func)

# Optimiser with gradient clipping

eta_max = 5e-5
eta_min = 1e-9
opt_state = Flux.setup(AdamW(eta = eta_max), model)

iters = 1000
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
        #floss(P.continuous_process, ŷ, X1_signed, 1.0f0)
    end
    Flux.update!(opt_state, model, ∇model)
    push!(losses, l)
   
    if i < 100 || i % 100 == 0
        @info "iter=$i loss=$(l) lr=$(lr)"
    end
end
println("Training complete")
# ---------------------------------------------------------------------
# Loss plot
# ---------------------------------------------------------------------
begin
    plt_loss = plot(1:length(losses), losses; 
        xlabel = "Iteration", ylabel = "Loss", 
        title = "Training Loss", legend = :none,
        linewidth = 1.5, color = :royalblue, background_color = :white,
        size = (900, 400))
    savefig(plt_loss, "loss.png")
    display(plt_loss)
    push!(all_plots, plt_loss)
end
println("Loss plot complete")
# ---------------------------------------------------------------------
# Marginal generation plot (100 samples) - run after training
# ---------------------------------------------------------------------
begin
    n_plot_samples = 1000
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
        plot!(plt, xcoords, ycoords; seriestype = :shape, c = :gray, a = 0.3, lc = :gray)
    end

    xlabel!(plt, "t")
    ylabel!(plt, "x")
    xlims!(plt, 0, x_right)
    ylims!(plt, y_min, y_max)
    title!(plt, "Marginal trajectories")
    savefig(plt, "marginal_generation.png")
    display(plt)
    push!(all_plots, plt)
end
println("Conditional generation plot complete")
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

    # Use analytical PDF from X1_continuous_distribution
    y_min_plot = minimum(y_final) - 0.2
    y_max_plot = maximum(y_final) + 0.2
    y_grid = collect(range(y_min_plot, y_max_plot; length = 200))
    
    # Evaluate the analytical density
    density_vals = pdf.(Ref(X1_continuous_distribution), y_grid)
    
    α = 0.25
    x_right = 1 + α * maximum(density_vals) * 1.25
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

    # Plot density curve instead of histogram
    density_x = 1 .+ α .* density_vals
    plot!(plt, density_x, y_grid; color = :gray, alpha = 0.7, linewidth = 2, fillrange = 1, fillalpha = 0.3, fillcolor = :gray)

    xlabel!(plt, "t")
    ylabel!(plt, "x")
    xlims!(plt, 0, x_right)
    ylims!(plt, y_min, y_max)
    title!(plt, "Conditional trajectories (endpoint-conditioned)")
    savefig(plt, "conditional_generation.png")
    display(plt)
    push!(all_plots, plt)
end
println("Final-time density check complete")
# ---------------------------------------------------------------------
# Final-time density check: model vs target X1 (using model-generated samples)
# ---------------------------------------------------------------------

     n_hist_samples = 5000
     X0_hist = ContinuousState(T.(rand(X0_continuous_distribution,1, n_hist_samples)))
     samples_hist = gen(Deterministic(), X0_hist, model, 0f0:0.005f0:1f0)
     final_hist = vec(Float64.(tensor(samples_hist)))

     x_min = minimum(final_hist)
     x_max = maximum(final_hist)
     xgrid = collect(range(x_min, x_max; length = 400))
     pdf_vals = pdf.(Ref(X1_continuous_distribution), xgrid)

     plt = plot(xgrid, pdf_vals; color = :black, linewidth = 2, label = "target pdf", background_color = :white)
    histogram!(plt, final_hist; normalize = :pdf, nbins = 200, color = :seagreen, alpha = 0.45, label = "model samples")
    xlabel!(plt, "x")
    ylabel!(plt, "density")
    title!(plt, "Model final-time density vs target")
    savefig(plt, "conditional_density_compare.png")
    display(plt)
    push!(all_plots, plt)

println("Aggregate plots complete")
# ---------------------------------------------------------------------
# Save raw data for plotting
# ---------------------------------------------------------------------
begin
    using DelimitedFiles
    
    # Create data directory if it doesn't exist
    data_dir = "plot_data"
    mkpath(data_dir)
    
    # 1. Save training losses
    writedlm(joinpath(data_dir, "losses.csv"), 
             hcat(1:length(losses), losses), 
             ',')
    println("Saved losses to $(data_dir)/losses.csv")
    
    # 2. Save marginal generation trajectories
    # tvec is (nsteps,), xttraj is (xdim, nsamples, nsteps)
    # Save time vector
    writedlm(joinpath(data_dir, "marginal_tvec.csv"), tvec, ',')
    
    # Save trajectories (reshape to 2D: each row is a trajectory over time)
    xttraj_2d = reshape(xttraj, size(xttraj, 1) * size(xttraj, 2), size(xttraj, 3))'  # (nsteps, nsamples)
    writedlm(joinpath(data_dir, "marginal_trajectories.csv"), xttraj_2d, ',')
    
    # Save initial conditions
    writedlm(joinpath(data_dir, "marginal_x0_vals.csv"), x0_vals, ',')
    
    # Save final samples
    final_samples = vec(Float64.(tensor(samples)))
    writedlm(joinpath(data_dir, "marginal_final_samples.csv"), final_samples, ',')
    println("Saved marginal generation data to $(data_dir)/marginal_*.csv")
    
    # 3. Save conditional trajectories
    writedlm(joinpath(data_dir, "conditional_tvec.csv"), tvec_ref[], ',')
    writedlm(joinpath(data_dir, "conditional_trajectories_Y.csv"), Y, ',')  # (n_cond_samples, n_timesteps)
    writedlm(joinpath(data_dir, "conditional_states_D.csv"), D, ',')  # discrete states
    writedlm(joinpath(data_dir, "conditional_y_final.csv"), y_final, ',')
    writedlm(joinpath(data_dir, "conditional_x1_targets.csv"), x1_targets, ',')
    
    # Save density curve for conditional plot
    writedlm(joinpath(data_dir, "conditional_density_ygrid.csv"), y_grid, ',')
    writedlm(joinpath(data_dir, "conditional_density_vals.csv"), density_vals, ',')
    println("Saved conditional trajectories data to $(data_dir)/conditional_*.csv")
    
    # 4. Save final density comparison data
    writedlm(joinpath(data_dir, "final_density_model_samples.csv"), final_hist, ',')
    writedlm(joinpath(data_dir, "final_density_xgrid.csv"), xgrid, ',')
    writedlm(joinpath(data_dir, "final_density_target_pdf.csv"), pdf_vals, ',')
    println("Saved final density comparison data to $(data_dir)/final_density_*.csv")
    
    # 5. Save metadata/info file
    open(joinpath(data_dir, "README.txt"), "w") do io
        println(io, "Raw data for plotting - Generated on $(now())")
        println(io, "="^60)
        println(io, "\n1. Training Data:")
        println(io, "   - losses.csv: [iteration, loss_value]")
        println(io, "   - Shape: $(length(losses)) iterations")
        println(io, "\n2. Marginal Generation Data:")
        println(io, "   - marginal_tvec.csv: time points ($(length(tvec)) steps)")
        println(io, "   - marginal_trajectories.csv: trajectories over time ($(size(xttraj_2d)))")
        println(io, "   - marginal_x0_vals.csv: initial conditions ($(length(x0_vals)) samples)")
        println(io, "   - marginal_final_samples.csv: final time values ($(length(final_samples)) samples)")
        println(io, "\n3. Conditional Trajectories Data:")
        println(io, "   - conditional_tvec.csv: time points ($(length(tvec_ref[])) steps)")
        println(io, "   - conditional_trajectories_Y.csv: continuous states ($(size(Y)))")
        println(io, "   - conditional_states_D.csv: discrete states ($(size(D)))")
        println(io, "   - conditional_y_final.csv: final values ($(length(y_final)) samples)")
        println(io, "   - conditional_x1_targets.csv: target endpoints ($(length(x1_targets)) samples)")
        println(io, "   - conditional_density_ygrid.csv: density plot y-axis ($(length(y_grid)) points)")
        println(io, "   - conditional_density_vals.csv: density values ($(length(density_vals)) points)")
        println(io, "\n4. Final Density Comparison:")
        println(io, "   - final_density_model_samples.csv: model samples ($(length(final_hist)) samples)")
        println(io, "   - final_density_xgrid.csv: x-axis for PDF ($(length(xgrid)) points)")
        println(io, "   - final_density_target_pdf.csv: target PDF values ($(length(pdf_vals)) points)")
        println(io, "\n5. Model Parameters:")
        println(io, "   - n_samples: $(n_samples)")
        println(io, "   - iterations: $(iters)")
        println(io, "   - embedding dimension: 1024")
        println(io, "   - n_plot_samples: $(n_plot_samples)")
        println(io, "   - n_cond_samples: $(n_cond_samples)")
    end
    println("Saved metadata to $(data_dir)/README.txt")
    println("\nAll raw data saved to '$(data_dir)/' directory")
end

# ---------------------------------------------------------------------
# Aggregate plots into a single PDF
# ---------------------------------------------------------------------
begin
    if !isempty(all_plots)
        combined = plot(all_plots...; layout = (length(all_plots), 1), size = (900, 600 * length(all_plots)), background_color = :white)
        savefig(combined, "all_plots.pdf")
    end
end
println("All plots complete")