using LinearAlgebra
using ForwardDiff
using Distributions, Random
using BenchmarkTools
using Plots, LaTeXStrings
# We will first attempt to implement a linear regression version of normalising flow, that is the link function is linear

function likelihood_function(θ, x, transformation)
    ȷ = ForwardDiff.jacobian(y -> transformation(θ, y), x)
    transformed_x = transformation(θ, x)
    return 1/2 * (transformed_x' * transformed_x) - logabsdet(ȷ)[1]
end

# We define the transformation as a multiple linear regression problem
function transformation(θ, x) 
    dimension = length(x)
    return reshape(θ, (dimension, dimension)) * x 
end

function train_parameters(X, transformation, parameter_dimension; learning_rate = 1e-4, n_iterations = 1000, minibatch_inclusion_probability = 0.01)
    θ = rand(Normal(0,1),parameter_dimension)
    θs = []
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
    return θs
end

function generate_data(n_samples::Int)
    X = zeros(2, n_samples)
    x1 = rand(Normal(0,1),n_samples)
    x2 = rand(Normal(0,1),n_samples)
    X[1,:] = x1 + x2
    X[2,:] = x1 
    return [X[:, i] for i in 1:n_samples]
end

function transform_all(θ, X, transformation)
    return [transformation(θ, x) for x in X]
end

# Function to create plots of the data distribution at different stages
function plot_transformation_evolution(X, θs, transformation, filename="flow_evolution.gif"; plot_every=1, fps=30)
    anim = @animate for i in 1:plot_every:length(θs)
    θ = θs[i]
    transformed_X = transform_all(θ, X, transformation)

    # Extract x and y components
    transformed_x = [point[1] for point in transformed_X]
    transformed_y = [point[2] for point in transformed_X]

    # Original data points
    original_x = [point[1] for point in X]
    original_y = [point[2] for point in X]

    # Create a two-panel plot
    p = plot(layout=(1, 2), size=(1000, 500), 
    title=["Original Data" "Transformed Data: Iteration $(i == 1 ? 0 : (i-1)*plot_every)"])

    # Plot original data
    scatter!(p[1], original_x, original_y, 
    markersize=3, alpha=0.6, 
    xlabel="x₁", ylabel="x₂", 
    legend=false)

    # Plot transformed data
    scatter!(p[2], transformed_x, transformed_y, 
    markersize=3, alpha=0.6, 
    xlabel="z₁", ylabel="z₂", 
    legend=false)

    # Calculate and display a contour of standard normal for reference
    if i > 1  # Skip for initialization
        x_range = range(-4, 4, length=100)
        y_range = range(-4, 4, length=100)
        z = [pdf(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), [x,y]) for y in y_range, x in x_range]
        contour!(p[2], x_range, y_range, z, levels=5, color=:gray, linestyle=:dash, alpha=0.5)
    end

    # Show the current transformation matrix
            matrix_θ = reshape(θ, (2, 2))

    end

    # Save as GIF
    gif(anim, filename, fps=fps)
    println("GIF saved as $filename")

    return anim
end

# Generate some correlated data
Random.seed!(42)  # For reproducibility
X = generate_data(250)

# Train the model and save parameters at intervals
n_iterations = 750  # Set a reasonable number of iterations for visualization
learning_rate = 0.005  # Higher learning rate for faster convergence in the visualization

@time θs = train_parameters(X, transformation, 4; 
    n_iterations=n_iterations, 
    learning_rate=learning_rate, 
    minibatch_inclusion_probability=0.05)

# Create the animation
plot_transformation_evolution(X, θs, transformation, "normalizing_flow_evolution.gif"; 
plot_every=1, fps=60)