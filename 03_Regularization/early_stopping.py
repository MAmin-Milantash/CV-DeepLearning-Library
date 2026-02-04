import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


if __name__ == "__main__":
    early_stopping = EarlyStopping(patience=3)

    val_losses = [1.0, 0.9, 0.85, 0.86, 0.88, 0.9]

    for epoch, loss in enumerate(val_losses):
        early_stopping.step(loss)
        print(f"Epoch {epoch}, val_loss={loss}")

        if early_stopping.should_stop:
            print("Early stopping triggered!")
            break

"""
📌 EarlyStopping is one of the most powerful regularizations in practice.
"""