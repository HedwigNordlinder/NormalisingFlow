using LinearAlgebra
using ForwardDiff
using Distributions, Random
using BenchmarkTools
using Plots, LaTeXStrings

include("utils.jl")

# We will first attempt to implement a linear regression version of normalising flow, that is the link function is linear

function likelihood_function(θ, x, transformation)
    ȷ = ForwardDiff.jacobian(y -> transformation(θ, y), x)
    transformed_x = transformation(θ, x)
    return 1/2 * (transformed_x' * transformed_x) - logabsdet(ȷ)[1]
end

function sigmoid(x)
    exp_x = exp.(x)
    return exp_x ./ (1 .+ exp_x)
end
function logit(x)
    return log.(x ./ (1 .- x))
end

function linear_sigmoid_transformation(θ, x) 
    dimension = length(x)
    return sigmoid(reshape(θ, (dimension, dimension)) * x) 
end

function inverse_linear_sigmoid_transformation(θ, x)
    dimension = length(x)
    transformation_matrix = inv(reshape(θ, (dimension, dimension)))
    return transformation_matrix * logit(x)
end
function example_network(θ, x)
    dimension = length(x)
    layer_one_theta = θ[1:dimension^2]
    layer_two_theta = θ[(dimension^2 + 1):end]
    return linear_sigmoid_transformation(layer_two_theta, linear_sigmoid_transformation(layer_one_theta, x))
end

function inverse_example_network(θ, x)
    dimension = length(x)
    layer_one_theta = θ[1:dimension^2]
    layer_two_theta = θ[(1:dimension^2 + 1):end]
    return inverse_linear_sigmoid_transformation(layer_two_theta, 
                                                inverse_linear_sigmoid_transformation(layer_one_theta, x))
end
function train_parameters(X, transformation, parameter_dimension; learning_rate = 1e-4, n_iterations = 1000, minibatch_inclusion_probability = 0.01)
    θ = rand(Normal(0,1),parameter_dimension)
    θs = []
    loss = []
    for _ in 1:n_iterations
        push!(θs, θ)
        minibatch_components = (rand(length(X)) .< minibatch_inclusion_probability)
        minibatch = X[minibatch_components]    
        ∇f = zeros(parameter_dimension)
        for s in minibatch
            f = y -> likelihood_function(y, s, transformation)
            ∇f = ∇f + ForwardDiff.gradient(f, θ)/length(minibatch)
        end 
        θ = θ - learning_rate * ∇f
    end
    return θs, loss
end


function visualize_flow_transformation(θ)
    # Create a simple grid in latent space (2D for visualization purposes)
    x_range = range(-3, 3, length=15)
    y_range = range(-3, 3, length=15)
    
    # Store original and transformed points
    latent_points_x = Float64[]
    latent_points_y = Float64[]
    transformed_points_x = Float64[]
    transformed_points_y = Float64[]
    
    for x in x_range
        for y in y_range
            # Create a point in latent space
            z = [x, y]
            
            # Add to latent points collection
            push!(latent_points_x, x)
            push!(latent_points_y, y)
            
            # Transform point using the inverse network
            # First apply sigmoid to normalize to (0,1) range
            z_normalized = sigmoid.(z)
            
            # Handle potential errors in the transformation
            try
                # Apply your inverse transformation
                transformed = inverse_example_network(θ, z_normalized)
                push!(transformed_points_x, transformed[1])
                push!(transformed_points_y, transformed[2])
            catch e
                # If the transformation fails, just skip this point
                println("Skipping point ($x, $y) - Error: $e")
            end
        end
    end
    
    # Create the plots
    p1 = scatter(
        latent_points_x, latent_points_y,
        title="Latent Space",
        xlabel="Z₁", ylabel="Z₂",
        markersize=3, color=:blue, alpha=0.6,
        legend=false
    )
    
    p2 = scatter(
        transformed_points_x, transformed_points_y,
        title="Transformed Space",
        xlabel="X₁", ylabel="X₂",
        markersize=3, color=:red, alpha=0.6,
        legend=false
    )
    
    # Plot original data if available
    if @isdefined(X) && @isdefined(labels)
        original_x = [point[1] for point in X]
        original_y = [point[2] for point in X]
        scatter!(p2, original_x, original_y, color=:green, alpha=0.4, markersize=2)
    end
    
    # Combine the plots
    combined_plot = plot(p1, p2, layout=(1,2), size=(1000, 500))
    
    return combined_plot
end

# Assuming you have theta from your model
# Replace this with your actual trained parameters
# θ = θs[end]  # Last set of parameters from training

# Generate/load data for visualization
n_samples = 250
X, labels, params = generate_data(n_samples)

# Train model or use existing parameters
n_iterations = 100  # Reduced for faster execution
learning_rate = 0.005
minibatch_inclusion_probability = 0.05

θs, _ = train_parameters(X, example_network, 8; 
    n_iterations=n_iterations, 
    learning_rate=learning_rate, 
    minibatch_inclusion_probability=minibatch_inclusion_probability)

# Get the final parameters
θ_final = θs[end]

# Create visualization
p = visualize_flow_transformation(θ_final)

# Save the plot
savefig(p, "flow_transformation.png")
println("Visualization saved to flow_transformation.png")