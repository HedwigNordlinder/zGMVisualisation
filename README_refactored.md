# Refactored Visualization Scripts

This directory contains a modular, well-organized codebase for training and visualizing flow-based generative models.

## File Structure

```
VisualisationScript/
├── main_refactored.jl      # Main training script
├── serialisation.jl         # Data saving/loading functions
├── plotting.jl              # Plotting functions
├── plot_from_saved.jl       # Example: plot from saved CSV files
├── sexyanimation.jl         # Animation utilities
└── plot_data/               # Directory for saved CSV data (created automatically)
    ├── losses.csv
    ├── marginal_*.csv
    ├── conditional_*.csv
    └── final_density_*.csv
```

## Quick Start

### 1. Train the model and generate all visualizations

```julia
include("main_refactored.jl")
```

This will:
- Train the model
- Save all raw data to `plot_data/` directory
- Generate and save individual plots as PNG files
- Create a combined PDF with all plots

### 2. Regenerate plots from saved data

After training, you can regenerate plots without retraining:

```julia
include("plot_from_saved.jl")
```

This demonstrates how to load data from CSV files and create plots.

## Module Documentation

### `serialisation.jl`

Functions for saving and loading data:

**Saving:**
- `save_training_data(losses; data_dir="plot_data")`
- `save_marginal_generation_data(tvec, xttraj, x0_vals, final_samples; data_dir="plot_data")`
- `save_conditional_trajectories_data(tvec, Y, D, y_final, x1_targets, y_grid, density_vals; data_dir="plot_data")`
- `save_final_density_data(model_samples, xgrid, pdf_vals; data_dir="plot_data")`

**Loading:**
- `load_training_data(data_dir="plot_data")` → returns `(iterations, losses)`
- `load_marginal_generation_data(data_dir="plot_data")` → returns `(tvec, trajectories, x0_vals, final_samples)`
- `load_conditional_trajectories_data(data_dir="plot_data")` → returns `(tvec, Y, D, y_final, x1_targets, y_grid, density_vals)`
- `load_final_density_data(data_dir="plot_data")` → returns `(model_samples, xgrid, pdf_vals)`

### `plotting.jl`

Each plotting function can accept either:
1. Raw data as arguments
2. No arguments (loads from `data_dir`)

**Available plots:**

```julia
# Plot training loss
plt = plot_loss(losses)                    # from raw data
plt = plot_loss(; data_dir="plot_data")    # from saved files

# Plot marginal generation
plt = plot_marginal_generation(tvec, trajectories, x0_vals, final_samples; target_dist=dist)
plt = plot_marginal_generation(; data_dir="plot_data", target_dist=dist)

# Plot conditional trajectories
plt = plot_conditional_trajectories(tvec, Y, D, y_final, x1_targets, y_grid, density_vals)
plt = plot_conditional_trajectories(; data_dir="plot_data")

# Plot final density comparison
plt = plot_final_density(model_samples, xgrid, pdf_vals)
plt = plot_final_density(; data_dir="plot_data")

# Save combined PDF
save_all_plots(plots_vector, "output.pdf")
```

## Usage Examples

### Example 1: Custom plotting from raw data

```julia
using Plots, Distributions
include("plotting.jl")

# Your data
losses = [1.2, 1.0, 0.8, 0.6]

# Create plot
plt = plot_loss(losses)
savefig(plt, "my_loss.png")
```

### Example 2: Load and modify plots

```julia
include("serialisation.jl")
include("plotting.jl")

# Load data
iters, losses = load_training_data("plot_data")

# Create custom plot
plt = plot_loss(losses)
plot!(plt; yscale=:log10, title="Loss (log scale)")
savefig(plt, "loss_log.png")
```

### Example 3: Batch processing

```julia
include("plotting.jl")

# Generate all plots from saved data
data_dir = "plot_data"
target_dist = MixtureModel([Normal(-0.5,0.1), Normal(0.6,0.5), Normal(0.9,0.1)], [0.2, 0.4, 0.4])

plots = [
    plot_loss(; data_dir=data_dir),
    plot_marginal_generation(; data_dir=data_dir, target_dist=target_dist),
    plot_conditional_trajectories(; data_dir=data_dir),
    plot_final_density(; data_dir=data_dir)
]

save_all_plots(plots, "all_results.pdf")
```

## Data Format

All data is saved in CSV format for easy interoperability with other tools (Python, R, MATLAB, etc.).

### Training data
- `losses.csv`: [iteration, loss_value]

### Marginal generation data
- `marginal_tvec.csv`: Time points (vector)
- `marginal_trajectories.csv`: Trajectories matrix (nsteps × nsamples)
- `marginal_x0_vals.csv`: Initial conditions (vector)
- `marginal_final_samples.csv`: Final values (vector)

### Conditional trajectories data
- `conditional_tvec.csv`: Time points (vector)
- `conditional_trajectories_Y.csv`: Continuous states (n_samples × n_timesteps)
- `conditional_states_D.csv`: Discrete states (n_samples × n_timesteps)
- `conditional_y_final.csv`: Final values (vector)
- `conditional_x1_targets.csv`: Target endpoints (vector)
- `conditional_density_ygrid.csv`: Density y-axis (vector)
- `conditional_density_vals.csv`: Density values (vector)

### Final density data
- `final_density_model_samples.csv`: Model samples (vector)
- `final_density_xgrid.csv`: PDF x-axis (vector)
- `final_density_target_pdf.csv`: Target PDF values (vector)

## Benefits of This Structure

1. **Modularity**: Separate concerns (training, plotting, data I/O)
2. **Reusability**: Functions can be used independently
3. **Testability**: Each module can be tested separately
4. **Flexibility**: Plot from raw data or saved files
5. **Interoperability**: CSV format works with any tool
6. **Maintainability**: Clear organization and documentation

## Migration from Old Script

The old `main.jl` is still available. To switch to the refactored version:

1. Backup your old script if needed
2. Use `main_refactored.jl` instead of `main.jl`
3. All functionality is preserved, just better organized

You can gradually migrate by:
- Using plotting functions in your existing code
- Saving data with serialization functions
- Loading and replotting without retraining

