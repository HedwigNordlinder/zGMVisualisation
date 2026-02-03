"""
Script to regenerate plots from saved CSV data.

This demonstrates how to use the plotting functions with saved data files.
"""

using Pkg
Pkg.activate("/Users/hedwignordlinder/Documents/Code/Julia/Karolinska/zGMVisualisation/")
using Plots, Distributions

include("plotting.jl")

# ============================================================================
# Configuration
# ============================================================================

data_dir = "plot_data"

# Define the target distribution (needed for some plots)
X1_continuous_distribution = MixtureModel([Normal(-0.5,0.1), Normal(0.6,0.5), Normal(0.9,0.1)], [0.2, 0.4, 0.4])

# ============================================================================
# Generate All Plots from Saved Data
# ============================================================================

println("Loading data and generating plots from '$data_dir/'...")
all_plots = []

# Plot 1: Training loss
println("  - Loading loss data...")
plt_loss = plot_loss(; data_dir=data_dir)
savefig(plt_loss, "loss_from_saved.png")
display(plt_loss)
push!(all_plots, plt_loss)
println("  ✓ Loss plot generated")

# Plot 2: Marginal generation
println("  - Loading marginal generation data...")
plt_marginal = plot_marginal_generation(; data_dir=data_dir, target_dist=X1_continuous_distribution)
savefig(plt_marginal, "marginal_generation_from_saved.png")
display(plt_marginal)
push!(all_plots, plt_marginal)
println("  ✓ Marginal generation plot generated")

# Plot 3: Conditional trajectories
println("  - Loading conditional trajectories data...")
plt_conditional = plot_conditional_trajectories(; data_dir=data_dir)
savefig(plt_conditional, "conditional_generation_from_saved.png")
display(plt_conditional)
push!(all_plots, plt_conditional)
println("  ✓ Conditional trajectories plot generated")

# Plot 4: Final density comparison
println("  - Loading final density data...")
plt_density = plot_final_density(; data_dir=data_dir)
savefig(plt_density, "conditional_density_compare_from_saved.png")
display(plt_density)
push!(all_plots, plt_density)
println("  ✓ Final density plot generated")

# Save combined PDF
save_all_plots(all_plots, "all_plots_from_saved.pdf")

println("\n" * "="^60)
println("All plots regenerated from saved data!")
println("  - Individual plots saved with '_from_saved' suffix")
println("  - Combined plots saved to 'all_plots_from_saved.pdf'")
println("="^60)

