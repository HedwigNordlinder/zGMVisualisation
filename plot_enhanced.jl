using Pkg
Pkg.activate("/Users/hedwignordlinder/Documents/Code/Julia/Karolinska/zGMVisualisation/")
using Plots, DelimitedFiles, Distributions

# ============================================================
# USER CONFIGURATION
# ============================================================
REGENERATE_CONDITIONAL_SAMPLES = false  # Set to true to re-draw conditional samples
N_COND_SAMPLES = 100  # Number of conditional samples (only used if regenerating)
# ============================================================

# Directory with saved data - get absolute path
script_dir = @__DIR__
data_dir = joinpath(dirname(script_dir), "plot_data")

println("Loading data from CSVs...")
all_plots = []

# ---------------------------------------------------------------------
# 1. Loss plot
# ---------------------------------------------------------------------
begin
    losses_data = readdlm(joinpath(data_dir, "losses.csv"), ',')
    iterations = losses_data[:, 1]
    losses = losses_data[:, 2]
    
    plt_loss = plot(iterations, losses; 
        xlabel = "Iteration", ylabel = "Loss", 
        title = "Training Loss", legend = :none,
        linewidth = 1.5, color = :royalblue, background_color = :white,
        size = (900, 400))
    
    display(plt_loss)
    push!(all_plots, plt_loss)
    println("✓ Loss plot recreated")
end

# ---------------------------------------------------------------------
# 2. Marginal generation plot
# ---------------------------------------------------------------------
begin
    tvec = vec(readdlm(joinpath(data_dir, "marginal_tvec.csv"), ','))
    xttraj_2d = readdlm(joinpath(data_dir, "marginal_trajectories.csv"), ',')
    x0_vals = vec(readdlm(joinpath(data_dir, "marginal_x0_vals.csv"), ','))
    final_samples = vec(readdlm(joinpath(data_dir, "marginal_final_samples.csv"), ','))
    
    # xttraj_2d is (nsteps, nsamples) - transpose to get trajectories as rows
    nsteps, nsamples = size(xttraj_2d)
    
    # Append final time point
    t_full = [tvec; 1.0]
    Y = xttraj_2d'  # Now (nsamples, nsteps)
    
    # Add final samples
    Y_full = hcat(Y, final_samples)  # (nsamples, nsteps+1)
    
    # Histogram of final-time values
    final_vals = final_samples
    nbins = 35
    edges = collect(range(minimum(final_vals), maximum(final_vals); length = nbins + 1))
    binwidth = edges[2] - edges[1]
    densities = zeros(Float64, nbins)
    for i in 1:nbins
        lo, hi = edges[i], edges[i + 1]
        densities[i] = count(x -> (x >= lo) && (x < hi), final_vals)
    end
    if !isempty(densities)
        densities ./= (length(final_vals) * binwidth)
    end
    
    α = 0.25
    x_right = 1 + α * (isempty(densities) ? 0 : maximum(densities)) * 1.25
    y_min = minimum(Y_full)
    y_max = maximum(Y_full)
    
    plt = plot(; legend = :none, size = (900, 600), background_color = :white)
    for i in 1:nsamples
        plot!(plt, t_full, Y_full[i, :]; color = :royalblue, alpha = 0.2, linewidth = 1)
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
    
    display(plt)
    push!(all_plots, plt)
    println("✓ Marginal generation plot recreated")
end

# ---------------------------------------------------------------------
# 3. ENHANCED Conditional generation plot with highlighted top-10 switchers
# ---------------------------------------------------------------------
begin
    if REGENERATE_CONDITIONAL_SAMPLES
        println("Regenerating conditional samples...")
        using ForwardBackward, Flowfusion, Flux, RandomFeatureMaps
        include("main.jl")  # Load model, P, sampleX0, sampleX1
        
        T = Float32
        n_cond_samples = N_COND_SAMPLES
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
        end
        
        tvec_cond = tvec_ref[]
        n_timesteps = length(tvec_cond)
        
        # Generate density for plot
        y_final = Y[:, end]
        y_min_plot = minimum(y_final) - 0.2
        y_max_plot = maximum(y_final) + 0.2
        y_grid = collect(range(y_min_plot, y_max_plot; length = 200))
        density_vals = pdf.(Ref(X1_continuous_distribution), y_grid)
        
        println("✓ Generated $n_cond_samples new conditional samples")
    else
        tvec_cond = vec(readdlm(joinpath(data_dir, "conditional_tvec.csv"), ','))
        Y = readdlm(joinpath(data_dir, "conditional_trajectories_Y.csv"), ',')
        D = Int.(readdlm(joinpath(data_dir, "conditional_states_D.csv"), ','))
        y_grid = vec(readdlm(joinpath(data_dir, "conditional_density_ygrid.csv"), ','))
        density_vals = vec(readdlm(joinpath(data_dir, "conditional_density_vals.csv"), ','))
        
        n_cond_samples = size(Y, 1)
        n_timesteps = size(Y, 2)
        println("✓ Loaded conditional samples from CSV")
    end
    
    # Count switches for each trajectory
    switch_counts = zeros(Int, n_cond_samples)
    for i in 1:n_cond_samples
        for j in 1:(n_timesteps - 1)
            if D[i, j] != D[i, j+1]
                switch_counts[i] += 1
            end
        end
    end
    
    # Find the 10 tracks with the most switches
    top_switchers_indices = partialsort(1:n_cond_samples, 1:min(10, n_cond_samples), 
                                        by=i -> switch_counts[i], rev=true)
    
    println("\n🎯 Top 10 switchers:")
    for (rank, idx) in enumerate(top_switchers_indices)
        println("   Rank $rank: Track $idx with $(switch_counts[idx]) switches")
    end
    println()
    
    # Define colors for discrete states
    state_colors = Dict(1 => :seagreen, 2 => :red)
    
    # Setup plot parameters
    α = 0.25
    x_right = 1 + α * maximum(density_vals) * 1.25
    y_min = minimum(Y)
    y_max = maximum(Y)
    
    # Create the plot
    plt = plot(; legend = :none, size = (900, 600), background_color = :white)
    
    # First pass: Plot background tracks (low opacity)
    for i in 1:n_cond_samples
        if i ∉ top_switchers_indices
            for j in 1:(n_timesteps - 1)
                seg_color = get(state_colors, D[i, j], :black)
                plot!(plt, tvec_cond[j:j+1], Y[i, j:j+1]; 
                      color = seg_color, alpha = 0.08, linewidth = 0.8)
            end
        end
    end
    
    # Second pass: Plot highlighted tracks (high opacity, thicker lines)
    for i in top_switchers_indices
        for j in 1:(n_timesteps - 1)
            seg_color = get(state_colors, D[i, j], :black)
            plot!(plt, tvec_cond[j:j+1], Y[i, j:j+1]; 
                  color = seg_color, alpha = 0.7, linewidth = 2.0)
        end
    end
    
    # Plot density curve
    density_x = 1 .+ α .* density_vals
    plot!(plt, density_x, y_grid; color = :gray, alpha = 0.7, linewidth = 2, 
          fillrange = 1, fillalpha = 0.3, fillcolor = :gray)
    
    xlabel!(plt, "t")
    ylabel!(plt, "x")
    xlims!(plt, 0, x_right)
    ylims!(plt, y_min, y_max)
    title!(plt, "Conditional trajectories (top 10 switchers highlighted)")
    
    savefig(plt, joinpath(dirname(script_dir), "conditional_generation_enhanced.png"))
    display(plt)
    push!(all_plots, plt)
    println("✓ Enhanced conditional generation plot created")
end

# ---------------------------------------------------------------------
# 4. Final-time density check
# ---------------------------------------------------------------------
begin
    final_hist = vec(readdlm(joinpath(data_dir, "final_density_model_samples.csv"), ','))
    xgrid = vec(readdlm(joinpath(data_dir, "final_density_xgrid.csv"), ','))
    pdf_vals = vec(readdlm(joinpath(data_dir, "final_density_target_pdf.csv"), ','))
    
    plt = plot(xgrid, pdf_vals; color = :black, linewidth = 2, label = "target pdf", 
               background_color = :white)
    histogram!(plt, final_hist; normalize = :pdf, nbins = 200, color = :seagreen, 
               alpha = 0.45, label = "model samples")
    xlabel!(plt, "x")
    ylabel!(plt, "density")
    title!(plt, "Model final-time density vs target")
    
    display(plt)
    push!(all_plots, plt)
    println("✓ Final density comparison plot recreated")
end

# ---------------------------------------------------------------------
# 5. Aggregate plots into a single PDF
# ---------------------------------------------------------------------
begin
    if !isempty(all_plots)
        combined = plot(all_plots...; layout = (length(all_plots), 1), 
                       size = (900, 600 * length(all_plots)), background_color = :white)
        savefig(combined, joinpath(dirname(script_dir), "all_plots_enhanced.pdf"))
        println("\n✨ All plots saved to all_plots_enhanced.pdf")
    end
end

println("\n✅ Done! All plots recreated with enhanced conditional tracks visualization.")

