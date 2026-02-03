"""
Serialization and deserialization functions for saving/loading training and visualization data.
"""

using DelimitedFiles

"""
    save_training_data(losses::Vector; data_dir="plot_data")

Save training loss data to CSV file.
"""
function save_training_data(losses::Vector; data_dir="plot_data")
    mkpath(data_dir)
    writedlm(joinpath(data_dir, "losses.csv"), 
             hcat(1:length(losses), losses), 
             ',')
    println("Saved losses to $(data_dir)/losses.csv")
end

"""
    save_marginal_generation_data(tvec, xttraj, x0_vals, final_samples; data_dir="plot_data")

Save marginal generation trajectory data to CSV files.

# Arguments
- `tvec`: Time vector (nsteps,)
- `xttraj`: Trajectories (xdim, nsamples, nsteps)
- `x0_vals`: Initial conditions (nsamples,)
- `final_samples`: Final time values (nsamples,)
"""
function save_marginal_generation_data(tvec, xttraj, x0_vals, final_samples; data_dir="plot_data")
    mkpath(data_dir)
    
    # Save time vector
    writedlm(joinpath(data_dir, "marginal_tvec.csv"), tvec, ',')
    
    # Save trajectories (reshape to 2D: each row is a trajectory over time)
    xttraj_2d = reshape(xttraj, size(xttraj, 1) * size(xttraj, 2), size(xttraj, 3))'
    writedlm(joinpath(data_dir, "marginal_trajectories.csv"), xttraj_2d, ',')
    
    # Save initial conditions and final samples
    writedlm(joinpath(data_dir, "marginal_x0_vals.csv"), x0_vals, ',')
    writedlm(joinpath(data_dir, "marginal_final_samples.csv"), final_samples, ',')
    
    println("Saved marginal generation data to $(data_dir)/marginal_*.csv")
end

"""
    save_conditional_trajectories_data(tvec, Y, D, y_final, x1_targets, y_grid, density_vals; data_dir="plot_data")

Save conditional trajectories data to CSV files.

# Arguments
- `tvec`: Time vector
- `Y`: Continuous state trajectories (n_samples, n_timesteps)
- `D`: Discrete state trajectories (n_samples, n_timesteps)
- `y_final`: Final values (n_samples,)
- `x1_targets`: Target endpoints (n_samples,)
- `y_grid`: Grid for density plot
- `density_vals`: Density values at grid points
"""
function save_conditional_trajectories_data(tvec, Y, D, y_final, x1_targets, y_grid, density_vals; data_dir="plot_data")
    mkpath(data_dir)
    
    writedlm(joinpath(data_dir, "conditional_tvec.csv"), tvec, ',')
    writedlm(joinpath(data_dir, "conditional_trajectories_Y.csv"), Y, ',')
    writedlm(joinpath(data_dir, "conditional_states_D.csv"), D, ',')
    writedlm(joinpath(data_dir, "conditional_y_final.csv"), y_final, ',')
    writedlm(joinpath(data_dir, "conditional_x1_targets.csv"), x1_targets, ',')
    writedlm(joinpath(data_dir, "conditional_density_ygrid.csv"), y_grid, ',')
    writedlm(joinpath(data_dir, "conditional_density_vals.csv"), density_vals, ',')
    
    println("Saved conditional trajectories data to $(data_dir)/conditional_*.csv")
end

"""
    save_final_density_data(model_samples, xgrid, pdf_vals; data_dir="plot_data")

Save final density comparison data to CSV files.

# Arguments
- `model_samples`: Samples from the model
- `xgrid`: Grid for PDF evaluation
- `pdf_vals`: Target PDF values at grid points
"""
function save_final_density_data(model_samples, xgrid, pdf_vals; data_dir="plot_data")
    mkpath(data_dir)
    
    writedlm(joinpath(data_dir, "final_density_model_samples.csv"), model_samples, ',')
    writedlm(joinpath(data_dir, "final_density_xgrid.csv"), xgrid, ',')
    writedlm(joinpath(data_dir, "final_density_target_pdf.csv"), pdf_vals, ',')
    
    println("Saved final density comparison data to $(data_dir)/final_density_*.csv")
end

"""
    load_training_data(data_dir="plot_data")

Load training loss data from CSV file. Returns (iterations, losses).
"""
function load_training_data(data_dir="plot_data")
    data = readdlm(joinpath(data_dir, "losses.csv"), ',')
    return data[:, 1], data[:, 2]
end

"""
    load_marginal_generation_data(data_dir="plot_data")

Load marginal generation data from CSV files.
Returns (tvec, trajectories, x0_vals, final_samples).
"""
function load_marginal_generation_data(data_dir="plot_data")
    tvec = vec(readdlm(joinpath(data_dir, "marginal_tvec.csv"), ','))
    trajectories = readdlm(joinpath(data_dir, "marginal_trajectories.csv"), ',')
    x0_vals = vec(readdlm(joinpath(data_dir, "marginal_x0_vals.csv"), ','))
    final_samples = vec(readdlm(joinpath(data_dir, "marginal_final_samples.csv"), ','))
    return tvec, trajectories, x0_vals, final_samples
end

"""
    load_conditional_trajectories_data(data_dir="plot_data")

Load conditional trajectories data from CSV files.
Returns (tvec, Y, D, y_final, x1_targets, y_grid, density_vals).
"""
function load_conditional_trajectories_data(data_dir="plot_data")
    tvec = vec(readdlm(joinpath(data_dir, "conditional_tvec.csv"), ','))
    Y = readdlm(joinpath(data_dir, "conditional_trajectories_Y.csv"), ',')
    D = readdlm(joinpath(data_dir, "conditional_states_D.csv"), ',')
    y_final = vec(readdlm(joinpath(data_dir, "conditional_y_final.csv"), ','))
    x1_targets = vec(readdlm(joinpath(data_dir, "conditional_x1_targets.csv"), ','))
    y_grid = vec(readdlm(joinpath(data_dir, "conditional_density_ygrid.csv"), ','))
    density_vals = vec(readdlm(joinpath(data_dir, "conditional_density_vals.csv"), ','))
    return tvec, Y, D, y_final, x1_targets, y_grid, density_vals
end

"""
    load_final_density_data(data_dir="plot_data")

Load final density comparison data from CSV files.
Returns (model_samples, xgrid, pdf_vals).
"""
function load_final_density_data(data_dir="plot_data")
    model_samples = vec(readdlm(joinpath(data_dir, "final_density_model_samples.csv"), ','))
    xgrid = vec(readdlm(joinpath(data_dir, "final_density_xgrid.csv"), ','))
    pdf_vals = vec(readdlm(joinpath(data_dir, "final_density_target_pdf.csv"), ','))
    return model_samples, xgrid, pdf_vals
end

