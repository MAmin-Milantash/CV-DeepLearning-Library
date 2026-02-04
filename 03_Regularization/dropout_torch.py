import torch
import torch.nn as nn

class DropoutExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    model = DropoutExample()
    model.train()

    x = torch.randn(4, 10)
    out = model(x)
    print(out)

""" 📌 Dropout is only enabled in model.train(). It is automatically disabled in model.eval(). """