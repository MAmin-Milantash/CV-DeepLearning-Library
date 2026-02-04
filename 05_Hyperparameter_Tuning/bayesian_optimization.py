from skopt import gp_minimize

def bayesian_optimization(model_class, search_space, train_loader, val_loader, criterion, n_calls=20):
    def objective(params):
        param_dict = dict(zip(search_space.keys(), params))
        model = model_class(**param_dict)
        train_model(model, train_loader, criterion)
        val_loss = evaluate_model(model, val_loader, criterion)
        return val_loss

    space = list(search_space.values())
    result = gp_minimize(objective, space, n_calls=n_calls)
    best_params = dict(zip(search_space.keys(), result.x))
    best_score = result.fun
    return best_params, best_score
