using Plots
using Flowfusion

"""
    animate_tracker(tracker; sample_index=1, outpath="tracker_animation.gif", fps=30, nbins=40, α=0.2, markercolor=:blue, linecolor=:red, linealpha=0.35, markersize=6)

Create a GIF showing a dot traveling along a single trajectory extracted from a `Tracker`.
The full path is drawn underneath with low alpha, and a right-anchored histogram
at `t = 1` (as in `main.jl`) is included.

Arguments
- `tracker`: A Flowfusion `Tracker` filled by your simulation.

Keywords
- `sample_index`: Which sample trajectory to animate (1-based).
- `outpath`: File path for the saved GIF.
- `fps`: Frames per second for the GIF.
- `nbins`: Number of bins for the final-time histogram.
- `α`: Horizontal scaling for the histogram width at the far right.
- `markercolor`: Color of the moving dot.
- `linecolor`: Color of the full path line.
- `linealpha`: Transparency of the full path line.
- `markersize`: Size of the moving dot marker.

Returns
- The saved GIF file path as a `String`.
"""
function animate_tracker(tracker;
    sample_index::Integer = 1,
    outpath::AbstractString = "tracker_animation.gif",
    fps::Integer = 30,
    nbins::Integer = 40,
    α::Real = 0.2,
    markercolor = :blue,
    linecolor = :red,
    linealpha::Real = 0.35,
    markersize::Real = 6,
)
    # Extract stacked arrays from the tracker
    tvec = stack_tracker(tracker, :t)              # length = nsteps
    xttraj = stack_tracker(tracker, :xt)           # dims ~ (xdim, nsamples, nsteps)

    # Basic sanity checks
    @assert ndims(xttraj) == 3 "Expected xttraj to have dims (xdim, nsamples, nsteps)"
    xdim, nsamples, nsteps = size(xttraj)
    @assert 1 <= sample_index <= nsamples "sample_index out of bounds (1..$nsamples)"
    @assert length(tvec) == nsteps "t vector length mismatch with trajectory steps"

    # Select 1D coordinate to visualize (first dimension)
    y_path = vec(xttraj[1, sample_index, :])       # the animated trajectory (over time)

    # Histogram of final-time values across all samples (first dimension)
    final_vals = vec(xttraj[1, :, end])
    vmin = minimum(final_vals)
    vmax = maximum(final_vals)
    edges = collect(range(vmin, vmax; length = nbins + 1))
    binwidth = edges[2] - edges[1]
    densities = zeros(Float64, nbins)
    for i in 1:nbins
        lo = edges[i]
        hi = edges[i+1]
        densities[i] = count(x -> (x >= lo) && (x < hi), final_vals)
    end
    if !isempty(densities)
        densities ./= (length(final_vals) * binwidth)  # normalize to approximate pdf
    end

    # Axes limits and labels
    x_right = 1 + α * (isempty(densities) ? 0 : maximum(densities)) * 1.2
    y_min = min(minimum(y_path), minimum(final_vals))
    y_max = max(maximum(y_path), maximum(final_vals))

    # Prepare animation
    anim = Animation()
    for j in 1:nsteps
        # Base plot
        plt = plot(; legend = :none, size = (900, 600))

        # Full path (underlay)
        plot!(plt, tvec, y_path; color = linecolor, alpha = linealpha, label = :none)

        # Right-anchored histogram at t = 1
        for i in 1:nbins
            xcoords = [1, 1 + α * densities[i], 1 + α * densities[i], 1]
            ycoords = [edges[i], edges[i], edges[i+1], edges[i+1]]
            plot!(plt, xcoords, ycoords; seriestype = :shape, c = :gray, a = 0.35, lc = :gray, label = :none)
        end

        # Moving dot
        scatter!(plt, [tvec[j]], [y_path[j]];
            color = markercolor, markersize = markersize, label = :none)

        # Decorations
        xlabel!(plt, "t")
        ylabel!(plt, "x")
        xlims!(plt, 0, x_right)
        ylims!(plt, y_min, y_max)
        title!(plt, "Trajectory animation (sample = $sample_index)")

        frame(anim, plt)
    end

    gif(anim, outpath; fps = fps)
    return String(outpath)
end


