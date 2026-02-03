"""
Plotting functions for visualization of training results.

Each function can accept either raw data or a path to load from CSV files.
"""

using Plots
using Distributions
include("serialisation.jl")

"""
    plot_loss(losses; data_dir=nothing, kwargs...)
    plot_loss(; data_dir="plot_data", kwargs...)

Plot training loss curve.

# Arguments
- `losses`: Vector of loss values (optional if loading from file)
- `data_dir`: Directory to load data from (if `losses` not provided)
- `kwargs...`: Additional plotting arguments

# Returns
- Plots.Plot object
"""
function plot_loss(losses::Union{Vector, Nothing}=nothing; data_dir=nothing, kwargs...)
    # Load from file if data not provided
    if losses === nothing
        data_dir = isnothing(data_dir) ? "plot_data" : data_dir
        _, losses = load_training_data(data_dir)
    end
    
    plt = plot(1:length(losses), losses; 
        xlabel = "Iteration", 
        ylabel = "Loss", 
        title = "Training Loss", 
        legend = :none,
        linewidth = 1.5, 
        color = :royalblue, 
        background_color = :white,
        size = (900, 400),
        kwargs...)
    
    return plt
end

"""
    plot_marginal_generation(tvec, trajectories, x0_vals, final_samples, target_dist; kwargs...)
    plot_marginal_generation(; data_dir="plot_data", target_dist, kwargs...)

Plot marginal generation trajectories with histograms.

# Arguments
- `tvec`: Time vector
- `trajectories`: Trajectory matrix (nsteps, nsamples)
- `x0_vals`: Initial condition values
- `final_samples`: Final time samples
- `target_dist`: Target distribution for PDF comparison
- `data_dir`: Directory to load data from (if raw data not provided)
- `kwargs...`: Additional plotting arguments

# Returns
- Plots.Plot object
"""
function plot_marginal_generation(tvec=nothing, trajectories=nothing, x0_vals=nothing, 
                                  final_samples=nothing; data_dir=nothing, target_dist, kwargs...)
    # Load from file if data not provided
    if isnothing(tvec)
        data_dir = isnothing(data_dir) ? "plot_data" : data_dir
        tvec, trajectories, x0_vals, final_samples = load_marginal_generation_data(data_dir)
    end
    
    nsteps, nsamples = size(trajectories)
    
    # Create histogram for initial distribution
    nbins_x0 = 30
    edges_x0 = range(minimum(x0_vals), maximum(x0_vals); length = nbins_x0 + 1)
    hist_x0 = fit(Histogram, x0_vals, edges_x0)
    densities_x0 = hist_x0.weights ./ (length(x0_vals) * step(edges_x0))
    
    # Create histogram for final distribution
    nbins_x1 = 30
    edges_x1 = range(minimum(final_samples), maximum(final_samples); length = nbins_x1 + 1)
    hist_x1 = fit(Histogram, final_samples, edges_x1)
    densities_x1 = hist_x1.weights ./ (length(final_samples) * step(edges_x1))
    
    # Compute plot dimensions
    α = 0.15
    x_left = -α * maximum(densities_x0) * 1.25
    x_right = 1 + α * maximum(densities_x1) * 1.25
    y_min = min(minimum(x0_vals), minimum(final_samples))
    y_max = max(maximum(x0_vals), maximum(final_samples))
    
    # Create main plot
    plt = plot(; legend = :none, size = (900, 600), background_color = :white)
    
    # Plot trajectories
    for i in 1:nsamples
        plot!(plt, tvec, trajectories[:, i]; color = :seagreen, alpha = 0.07, linewidth = 0.8)
    end
    
    # Plot initial histogram
    for i in 1:nbins_x0
        xcoords = [-α * densities_x0[i], 0, 0, -α * densities_x0[i]]
        ycoords = [edges_x0[i], edges_x0[i], edges_x0[i+1], edges_x0[i+1]]
        plot!(plt, xcoords, ycoords; seriestype = :shape, c = :gray, a = 0.45, lc = :gray)
    end
    
    # Plot final histogram
    for i in 1:nbins_x1
        xcoords = [1, 1 + α * densities_x1[i], 1 + α * densities_x1[i], 1]
        ycoords = [edges_x1[i], edges_x1[i], edges_x1[i+1], edges_x1[i+1]]
        plot!(plt, xcoords, ycoords; seriestype = :shape, c = :gray, a = 0.45, lc = :gray)
    end
    
    # Plot target density
    y_target = range(y_min, y_max; length = 200)
    pdf_target = pdf.(Ref(target_dist), y_target)
    x_target = 1 .+ α .* pdf_target
    plot!(plt, x_target, y_target; color = :red, linewidth = 2.5, alpha = 0.8, linestyle = :dash)
    
    xlabel!(plt, "t")
    ylabel!(plt, "x")
    xlims!(plt, x_left, x_right)
    ylims!(plt, y_min, y_max)
    title!(plt, "Marginal generation ($(nsamples) samples)")
    
    return plt
end

"""
    plot_conditional_trajectories(tvec, Y, D, y_final, x1_targets, y_grid, density_vals; kwargs...)
    plot_conditional_trajectories(; data_dir="plot_data", kwargs...)

Plot conditional trajectories with density curve.

# Arguments
- `tvec`: Time vector
- `Y`: Continuous state trajectories (n_samples, n_timesteps)
- `D`: Discrete state trajectories (n_samples, n_timesteps)
- `y_final`: Final values
- `x1_targets`: Target endpoints
- `y_grid`: Grid for density plot
- `density_vals`: Density values at grid points
- `data_dir`: Directory to load data from (if raw data not provided)
- `kwargs...`: Additional plotting arguments

# Returns
- Plots.Plot object
"""
function plot_conditional_trajectories(tvec=nothing, Y=nothing, D=nothing, y_final=nothing, 
                                       x1_targets=nothing, y_grid=nothing, density_vals=nothing;
                                       data_dir=nothing, kwargs...)
    # Load from file if data not provided
    if isnothing(tvec)
        data_dir = isnothing(data_dir) ? "plot_data" : data_dir
        tvec, Y, D, y_final, x1_targets, y_grid, density_vals = load_conditional_trajectories_data(data_dir)
    end
    
    n_samples = size(Y, 1)
    
    α = 0.25
    x_right = 1 + α * maximum(density_vals) * 1.25
    y_min = min(minimum(Y), minimum(x1_targets))
    y_max = max(maximum(Y), maximum(x1_targets))
    
    plt = plot(; legend = :none, size = (900, 600), background_color = :white)
    state_colors = Dict(1 => :seagreen, 2 => :red)
    
    # Plot trajectories with color coding by state
    for i in 1:n_samples
        for j in 1:(length(tvec) - 1)
            seg_color = get(state_colors, D[i, j], :black)
            plot!(plt, tvec[j:j+1], Y[i, j:j+1]; color = seg_color, alpha = 0.25, linewidth = 1.2)
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
    title!(plt, "Conditional trajectories (endpoint-conditioned)")
    
    return plt
end

"""
    plot_final_density(model_samples, xgrid, pdf_vals; kwargs...)
    plot_final_density(; data_dir="plot_data", kwargs...)

Plot final density comparison between model and target.

# Arguments
- `model_samples`: Samples from the model
- `xgrid`: Grid for PDF evaluation
- `pdf_vals`: Target PDF values
- `data_dir`: Directory to load data from (if raw data not provided)
- `kwargs...`: Additional plotting arguments

# Returns
- Plots.Plot object
"""
function plot_final_density(model_samples=nothing, xgrid=nothing, pdf_vals=nothing; 
                           data_dir=nothing, kwargs...)
    # Load from file if data not provided
    if isnothing(model_samples)
        data_dir = isnothing(data_dir) ? "plot_data" : data_dir
        model_samples, xgrid, pdf_vals = load_final_density_data(data_dir)
    end
    
    plt = plot(xgrid, pdf_vals; 
               color = :black, 
               linewidth = 2, 
               label = "target pdf", 
               background_color = :white,
               size = (900, 600))
    
    histogram!(plt, model_samples; 
               normalize = :pdf, 
               nbins = 200, 
               color = :seagreen, 
               alpha = 0.45, 
               label = "model samples")
    
    xlabel!(plt, "x")
    ylabel!(plt, "density")
    title!(plt, "Model final-time density vs target")
    
    return plt
end

"""
    save_all_plots(plots::Vector, filename="all_plots.pdf"; kwargs...)

Combine multiple plots into a single PDF.

# Arguments
- `plots`: Vector of Plot objects
- `filename`: Output filename
- `kwargs...`: Additional plotting arguments
"""
function save_all_plots(plots::Vector, filename="all_plots.pdf"; kwargs...)
    if !isempty(plots)
        combined = plot(plots...; 
                       layout = (length(plots), 1), 
                       size = (900, 600 * length(plots)), 
                       background_color = :white,
                       kwargs...)
        savefig(combined, filename)
        println("Saved combined plots to $filename")
    end
end

