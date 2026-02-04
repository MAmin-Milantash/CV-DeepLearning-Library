from hyperparameter_search import grid_search, random_search, bayesian_search  # hypothetical import

# Define hyperparameter space
param_space = {
    "learning_rate": [0.001, 0.01],
    "batch_size": [32, 64],
    "hidden_dim": [64, 128]
}

# Run grid search
best_params_grid = grid_search(SimpleMLP, param_space, epochs=3)

# Run random search
best_params_random = random_search(SimpleMLP, param_space, n_iter=5, epochs=3)

# Run Bayesian search
best_params_bayes = bayesian_search(SimpleMLP, param_space, n_iter=5, epochs=3)