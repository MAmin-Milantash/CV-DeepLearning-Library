import random

def random_search(model_class, param_dist, n_iter, train_loader, val_loader, criterion):
    best_score = float('inf')
    best_params = None

    for _ in range(n_iter):
        params = {k: random.choice(v) for k, v in param_dist.items()}
        model = model_class(**params)
        train_model(model, train_loader, criterion)
        val_loss = evaluate_model(model, val_loader, criterion)

        if val_loss < best_score:
            best_score = val_loss
            best_params = params

    return best_params, best_score
