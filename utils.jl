using Distributions
using Random
using StatsPlots
using LinearAlgebra

"""
    generate_data(n_samples::Int; 
                 dim::Int=2, 
                 n_components::Int=3, 
                 weights=nothing, 
                 means=nothing, 
                 covs=nothing, 
                 seed=nothing)

Generate samples from a mixture of Gaussian distributions.

# Arguments
- `n_samples::Int`: Number of samples to generate
- `dim::Int=2`: Dimensionality of the data
- `n_components::Int=3`: Number of Gaussian components in the mixture
- `weights=nothing`: Optional vector of component weights (will be normalized)
- `means=nothing`: Optional array of component means (dim × n_components)
- `covs=nothing`: Optional array of covariance matrices (dim × dim × n_components)
- `seed=nothing`: Optional random seed for reproducibility

# Returns
- `samples`: Matrix of shape (dim, n_samples) containing the generated samples
- `component_labels`: Vector indicating which component generated each sample
- `mixture_params`: NamedTuple containing the parameters of the mixture model
"""
function generate_data(n_samples::Int; 
                      dim::Int=2, 
                      n_components::Int=3, 
                      weights=nothing, 
                      means=nothing, 
                      covs=nothing, 
                      seed=nothing)
    
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Set up weights if not provided
    if isnothing(weights)
        weights = ones(n_components) / n_components
    else
        weights = weights ./ sum(weights)  # Normalize weights
    end
    
    # Set up means if not provided
    if isnothing(means)
        # Create means that are reasonably separated
        means = 4.0 * (rand(dim, n_components) .- 0.5)
    end
    
    # Set up covariance matrices if not provided
    if isnothing(covs)
        covs = zeros(dim, dim, n_components)
        for i in 1:n_components
            # Create a random positive definite matrix for covariance
            A = rand(dim, dim)
            covs[:,:,i] = 0.5 * (A * A' + 0.1 * I)  # Ensure positive definiteness
        end
    end
    
    # Create component distributions
    components = [MvNormal(means[:,i], Symmetric(covs[:,:,i])) for i in 1:n_components]
    
    # Create mixture model
    mixture = MixtureModel(components, weights)
    
    # Generate samples as an array of arrays
    samples = Vector{Vector{Float64}}(undef, n_samples)
    component_labels = zeros(Int, n_samples)
    
    for i in 1:n_samples
        # Sample component
        component = rand(Categorical(weights))
        # Sample from component as a vector
        samples[i] = rand(components[component])
        component_labels[i] = component
    end
    
    # Return the generated samples, labels, and mixture parameters
    mixture_params = (
        weights = weights,
        means = means,
        covs = covs
    )
    
    return samples, component_labels, mixture_params
end

# Example usage:
# samples, labels, params = generate_data(1000, dim=2, n_components=3, seed=123)

# Visualization helper function
function visualize_mixture(samples, labels, mixture_params; show_contours=true)
    # Extract the first two dimensions for visualization
    dim = length(samples[1])
    
    if dim > 2
        println("Visualizing first two dimensions of data...")
        x_coords = [sample[1] for sample in samples]
        y_coords = [sample[2] for sample in samples]
    else
        x_coords = [sample[1] for sample in samples]
        y_coords = [sample[2] for sample in samples]
    end
    
    # Create scatter plot of samples colored by component
    p = scatter(x_coords, y_coords, 
                group=labels, 
                alpha=0.6, 
                legend=:topleft,
                title="Mixture of Gaussians",
                xlabel="x₁", 
                ylabel="x₂",
                markersize=3)
    
    # Optionally add contours to show density
    if show_contours && dim == 2
        x_range = range(minimum(x_coords) - 1, maximum(x_coords) + 1, length=100)
        y_range = range(minimum(y_coords) - 1, maximum(y_coords) + 1, length=100)
        
        for i in 1:length(mixture_params.weights)
            μ = mixture_params.means[:,i]
            Σ = mixture_params.covs[:,:,i]
            
            if size(μ, 1) > 2
                μ = μ[1:2]
                Σ = Σ[1:2, 1:2]
            end
            
            dist = MvNormal(μ, Symmetric(Σ))
            z = [pdf(dist, [x,y]) for y in y_range, x in x_range]
            contour!(p, x_range, y_range, z, levels=5, alpha=0.5, color=i)
        end
    end
    
    return p
end


# Function to visualize transformation from latent space to data space
function visualize_transformation(θ, latent_grid_range=(-3:0.2:3, -3:0.2:3); samples=nothing, labels=nothing, mixture_params=nothing)
    dimension = 2  # We'll work with 2D data for visualization
    
    # Create grid points in latent space
    latent_x = collect(latent_grid_range[1])
    latent_y = collect(latent_grid_range[2])
    
    # Initialize arrays to store transformed points
    n_points_x = length(latent_x)
    n_points_y = length(latent_y)
    transformed_points = Array{Vector{Float64}}(undef, n_points_x, n_points_y)
    
    # Transform each point in the latent grid
    for (i, x) in enumerate(latent_x)
        for (j, y) in enumerate(latent_y)
            latent_point = [x, y]
            
            # Apply inverse transformation to map from latent to data space
            try
                # We need to ensure the inverse transformation produces valid inputs
                # for the forward transformation (between 0 and 1 for sigmoid)
                # So we first apply sigmoid to the latent point
                normalized_point = sigmoid.(latent_point)
                
                # Then apply the inverse network
                transformed_points[i, j] = inverse_example_network(θ, normalized_point)
            catch e
                # Handle potential numerical issues
                transformed_points[i, j] = [NaN, NaN]
            end
        end
    end
    
    # Create plots
    latent_plot = scatter(title="Latent Space Grid", xlabel="Z₁", ylabel="Z₂")
    for i in 1:n_points_x
        for j in 1:n_points_y
            scatter!([latent_x[i]], [latent_y[j]], markersize=1, color=:blue, alpha=0.5, label="")
        end
    end
    
    # Extract x and y coordinates from transformed points
    transformed_x = [point[1] for point in transformed_points if !any(isnan, point)]
    transformed_y = [point[2] for point in transformed_points if !any(isnan, point)]
    
    # Plot transformed points
    data_plot = scatter(transformed_x, transformed_y, 
                        title="Transformed to Data Space", 
                        xlabel="X₁", ylabel="X₂", 
                        markersize=1, color=:red, alpha=0.5, label="Transformed Grid")
    
    # If original data samples are provided, plot them too
    if !isnothing(samples) && !isnothing(labels)
        x_coords = [sample[1] for sample in samples]
        y_coords = [sample[2] for sample in samples]
        
        scatter!(data_plot, x_coords, y_coords, 
                group=labels, 
                alpha=0.6, 
                markersize=3,
                label="Original Data")
        
        # Optionally add contours to show density
        if !isnothing(mixture_params) && dimension == 2
            x_range = range(minimum(x_coords) - 1, maximum(x_coords) + 1, length=100)
            y_range = range(minimum(y_coords) - 1, maximum(y_coords) + 1, length=100)
            
            for i in 1:length(mixture_params.weights)
                μ = mixture_params.means[:,i]
                Σ = mixture_params.covs[:,:,i]
                
                dist = MvNormal(μ, Symmetric(Σ))
                z = [pdf(dist, [x,y]) for y in y_range, x in x_range]
                contour!(data_plot, x_range, y_range, z, levels=5, alpha=0.3, color=i)
            end
        end
    end
    
    # Combine plots side by side
    combined_plot = plot(latent_plot, data_plot, layout=(1,2), size=(1000, 500))
    
    return combined_plot
end

# Function to visualize the evolution of the transformation
function visualize_transformation_evolution(θs, latent_grid_range=(-3:0.2:3, -3:0.2:3); 
                                        samples=nothing, labels=nothing, mixture_params=nothing,
                                        plot_every=10, fps=15, output_file="transformation_evolution.gif")
    n_frames = length(θs) ÷ plot_every
    animation = @animate for i in 1:plot_every:length(θs)
        println("Generating frame $(i÷plot_every) of $n_frames")
        visualize_transformation(θs[i], latent_grid_range; 
                                samples=samples, labels=labels, mixture_params=mixture_params)
    end
    
    gif(animation, output_file, fps=fps)
    println("Animation saved to $output_file")
end
