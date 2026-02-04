import itertools

def grid_search(model_class, param_grid, train_loader, val_loader, criterion):
    best_score = float('inf')
    best_params = None

    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        model = model_class(**params)
        train_model(model, train_loader, criterion)
        val_loss = evaluate_model(model, val_loader, criterion)

        if val_loss < best_score:
            best_score = val_loss
            best_params = params

    return best_params, best_score
