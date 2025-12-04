using Plots
using Flowfusion
using ForwardBackward

"""
    animate_tracker(tracker; sample_index=1, outpath="tracker_animation.gif", fps=30, nbins=40, α=0.2, markercolor=:blue, linecolor=:red, linealpha=0.35, markersize=6, speed=1.0)

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
    speed::Real = 1.0,
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

    # Prepare animation timeline (supports smooth slow-down/speed-up via interpolation)
    frame_positions = collect(1:speed:nsteps)  # may be fractional if speed < 1
    anim = Animation()
    for f in frame_positions
        j0 = clamp(floor(Int, f), 1, nsteps)
        j1 = clamp(j0 + 1, 1, nsteps)
        w = clamp(f - j0, 0.0, 1.0)
        tj = (1 - w) * tvec[j0] + w * tvec[j1]
        yj = (1 - w) * y_path[j0] + w * y_path[j1]
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
        scatter!(plt, [tj], [yj];
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

"""
    animate_tracker_multi(tracker; sample_indices=1:100, outpath="marginal_animation.gif", fps=30, nbins=40, α=0.2, markercolor=:blue, markersize=4, show_paths=true, linecolor=:red, linealpha=0.25, speed=1.0, path_sample_stride=1)

Animate many dots (samples) traveling across time using a batched Flowfusion `Tracker`.
Also draws a right-anchored histogram at t=1 built from the selected samples.
"""
function animate_tracker_multi(tracker;
    sample_indices = 1:100,
    outpath::AbstractString = "marginal_animation.gif",
    fps::Integer = 30,
    nbins::Integer = 40,
    α::Real = 0.2,
    markercolor = :blue,
    markersize::Real = 4,
    show_paths::Bool = true,
    linecolor = :red,
    linealpha::Real = 0.25,
    speed::Real = 1.0,
    path_sample_stride::Integer = 1,
)
    tvec = stack_tracker(tracker, :t)
    xttraj = stack_tracker(tracker, :xt)             # (xdim, nsamples, nsteps)
    @assert ndims(xttraj) == 3 "Expected xttraj to have dims (xdim, nsamples, nsteps)"
    xdim, nsamples, nsteps = size(xttraj)
    inds = collect(filter(i -> 1 <= i <= nsamples, sample_indices))
    @assert !isempty(inds) "No valid sample indices to animate."

    # Use only first dimension for y coordinate
    Y = Array{Float64}(undef, length(inds), nsteps)
    for (k, i) in enumerate(inds)
        Y[k, :] = vec(xttraj[1, i, :])
    end
    final_vals = vec(xttraj[1, inds, end])

    # Histogram
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
        densities ./= (length(final_vals) * binwidth)
    end

    x_right = 1 + α * (isempty(densities) ? 0 : maximum(densities)) * 1.2
    y_min = minimum(Y)
    y_max = maximum(Y)

    # Frame positions with interpolation for smooth speed control
    frame_positions = collect(1:speed:nsteps)
    anim = Animation()
    for f in frame_positions
        j0 = clamp(floor(Int, f), 1, nsteps)
        j1 = clamp(j0 + 1, 1, nsteps)
        w = clamp(f - j0, 0.0, 1.0)
        tj = (1 - w) * tvec[j0] + w * tvec[j1]
        plt = plot(; legend = :none, size = (900, 600))
        # Right-anchored histogram
        for i in 1:nbins
            xcoords = [1, 1 + α * densities[i], 1 + α * densities[i], 1]
            ycoords = [edges[i], edges[i], edges[i+1], edges[i+1]]
            plot!(plt, xcoords, ycoords; seriestype = :shape, c = :gray, a = 0.35, lc = :gray, label = :none)
        end
        # Underlying paths (optional)
        if show_paths
            for k in 1:path_sample_stride:size(Y, 1)
                plot!(plt, tvec, Y[k, :]; color = linecolor, alpha = linealpha, label = :none)
            end
        end
        # Moving dots
        yj = (1 - w) .* Y[:, j0] .+ w .* Y[:, j1]
        xcoords = fill(tj, size(Y, 1))
        scatter!(plt, xcoords, yj; color = markercolor, markersize = markersize, label = :none)
        xlabel!(plt, "t")
        ylabel!(plt, "x")
        xlims!(plt, 0, x_right)
        ylims!(plt, y_min, y_max)
        title!(plt, "Marginal animation: $(length(inds)) samples")
        frame(anim, plt)
    end
    gif(anim, outpath; fps = fps)
    return String(outpath)
end

"""
    animate_conditional_multi(X0_sw, X1_sw, P; sample_indices=1:100, δt=0.01, outpath="conditional_animation.gif", fps=30, nbins=40, α=0.2, markercolor=:blue, markersize=4, show_paths=true, linecolor=:red, linealpha=0.25, speed=1.0, path_sample_stride=1)

Generate endpoint-conditioned trajectories for the selected indices and animate many
dots moving across time simultaneously. Includes a right-anchored histogram at t=1.
"""
function animate_conditional_multi(X0_sw, X1_sw, P;
    sample_indices = 1:100,
    δt::Real = 0.01,
    outpath::AbstractString = "conditional_animation.gif",
    fps::Integer = 30,
    nbins::Integer = 40,
    α::Real = 0.2,
    markercolor = :blue,
    markersize::Real = 4,
    show_paths::Bool = true,
    linecolor = :red,
    linealpha::Real = 0.25,
    speed::Real = 1.0,
    path_sample_stride::Integer = 1,
)
    inds = collect(sample_indices)
    # Build per-sample trackers and gather into arrays
    tvec_ref = nothing
    Y = nothing
    y_final = Float64[]
    for (k, i) in enumerate(inds)
        cpaths = Tracker()
        # Slice out single-sample SwitchingStates (views)
        X0_i = ForwardBackward.SwitchingState(
            ForwardBackward.ContinuousState(view(X0_sw.continuous_state.state, 1:1, i:i)),
            ForwardBackward.DiscreteState(X0_sw.discrete_state.K, view(X0_sw.discrete_state.state, 1:1, i:i)),
        )
        X1_i = ForwardBackward.SwitchingState(
            ForwardBackward.ContinuousState(view(X1_sw.continuous_state.state, 1:1, i:i)),
            ForwardBackward.DiscreteState(X1_sw.discrete_state.K, view(X1_sw.discrete_state.state, 1:1, i:i)),
        )
        _ = ForwardBackward.endpoint_conditioned_sample(X1_i, X0_i, P, eltype(X0_sw.continuous_state.state)(1); δt = δt, tracker = cpaths)
        tvec_i = stack_tracker(cpaths, :t)
        xttraj_i = stack_tracker(cpaths, :xt; tuple_index = 1) # continuous component
        y_i = vec(xttraj_i[1, 1, :])
        if tvec_ref === nothing
            tvec_ref = tvec_i
            Y = Array{Float64}(undef, length(inds), length(tvec_ref))
        else
            @assert length(tvec_i) == length(tvec_ref) "All conditional trajectories must share the same time grid."
        end
        Y[k, :] = y_i
        push!(y_final, y_i[end])
    end
    @assert tvec_ref !== nothing

    # Histogram at t=1 from y_final
    vmin = minimum(y_final)
    vmax = maximum(y_final)
    edges = collect(range(vmin, vmax; length = nbins + 1))
    binwidth = edges[2] - edges[1]
    densities = zeros(Float64, nbins)
    for i in 1:nbins
        lo = edges[i]
        hi = edges[i+1]
        densities[i] = count(x -> (x >= lo) && (x < hi), y_final)
    end
    if !isempty(densities)
        densities ./= (length(y_final) * binwidth)
    end

    x_right = 1 + α * (isempty(densities) ? 0 : maximum(densities)) * 1.2
    y_min = minimum(Y)
    y_max = maximum(Y)

    # Build frame positions with interpolation
    nsteps = length(tvec_ref)
    frame_positions = collect(1:speed:nsteps)
    anim = Animation()
    for f in frame_positions
        j0 = clamp(floor(Int, f), 1, nsteps)
        j1 = clamp(j0 + 1, 1, nsteps)
        w = clamp(f - j0, 0.0, 1.0)
        tj = (1 - w) * tvec_ref[j0] + w * tvec_ref[j1]
        plt = plot(; legend = :none, size = (900, 600))
        # Right-anchored histogram
        for i in 1:nbins
            xcoords = [1, 1 + α * densities[i], 1 + α * densities[i], 1]
            ycoords = [edges[i], edges[i], edges[i+1], edges[i+1]]
            plot!(plt, xcoords, ycoords; seriestype = :shape, c = :gray, a = 0.35, lc = :gray, label = :none)
        end
        # Underlying paths (optional)
        if show_paths
            for k in 1:path_sample_stride:size(Y, 1)
                plot!(plt, tvec_ref, Y[k, :]; color = linecolor, alpha = linealpha, label = :none)
            end
        end
        # Moving dots
        yj = (1 - w) .* Y[:, j0] .+ w .* Y[:, j1]
        xcoords = fill(tj, size(Y, 1))
        scatter!(plt, xcoords, yj; color = markercolor, markersize = markersize, label = :none)
        xlabel!(plt, "t")
        ylabel!(plt, "x")
        xlims!(plt, 0, x_right)
        ylims!(plt, y_min, y_max)
        title!(plt, "Conditional animation: $(length(inds)) samples")
        frame(anim, plt)
    end
    gif(anim, outpath; fps = fps)
    return String(outpath)
end


