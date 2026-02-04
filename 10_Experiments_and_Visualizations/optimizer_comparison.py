import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def compare_optimizers(model_class, optimizers_dict, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    for name, optimizer_func in optimizers_dict.items():
        model = model_class()
        optimizer = optimizer_func(model.parameters())
        losses = []
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
        results[name] = losses
        print(f"{name} final loss: {losses[-1]:.4f}")
    
    # Plot
    for name, loss_list in results.items():
        plt.plot(loss_list, label=name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Example usage
# optim_dict = {
#     "SGD": lambda params: optim.SGD(params, lr=0.01),
#     "Adam": lambda params: optim.Adam(params, lr=0.001)
# }
# compare_optimizers(SimpleMLP, optim_dict, train_loader, val_loader)