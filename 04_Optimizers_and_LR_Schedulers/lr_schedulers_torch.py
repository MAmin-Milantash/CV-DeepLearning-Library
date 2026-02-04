import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

# =====================================
# Optimizer Setup (Demo)
# =====================================
# فرض کنید یک مدل ساده داریم
import torch.nn as nn

model = nn.Linear(10, 1)  # نمونه ساده

# =====================================
# 1️⃣ SGD + StepLR
# =====================================
optimizer_sgd = SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler_step = StepLR(optimizer_sgd, step_size=5, gamma=0.5)

# =====================================
# 2️⃣ Adam + ExponentialLR
# =====================================
optimizer_adam = Adam(model.parameters(), lr=0.01)
scheduler_exp = ExponentialLR(optimizer_adam, gamma=0.9)

# =====================================
# 3️⃣ Cosine Annealing
# =====================================
optimizer_cos = SGD(model.parameters(), lr=0.1)
scheduler_cos = CosineAnnealingLR(optimizer_cos, T_max=10, eta_min=0.001)

# =====================================
# 4️⃣ ReduceLROnPlateau (monitors validation loss)
# =====================================
optimizer_plateau = Adam(model.parameters(), lr=0.01)
scheduler_plateau = ReduceLROnPlateau(optimizer_plateau, mode='min', factor=0.5, patience=3, verbose=True)

# =====================================
# Demo / Step through schedulers
# =====================================
steps = 15
print("=== StepLR Demo ===")
for step in range(steps):
    # فرض کنید یک step train داریم
    optimizer_sgd.step()
    scheduler_step.step()
    print(f"Step {step + 1}: LR = {optimizer_sgd.param_groups[0]['lr']:.5f}")

print("\n=== ExponentialLR Demo ===")
for step in range(steps):
    optimizer_adam.step()
    scheduler_exp.step()
    print(f"Step {step + 1}: LR = {optimizer_adam.param_groups[0]['lr']:.5f}")

print("\n=== CosineAnnealingLR Demo ===")
for step in range(steps):
    optimizer_cos.step()
    scheduler_cos.step()
    print(f"Step {step + 1}: LR = {optimizer_cos.param_groups[0]['lr']:.5f}")

print("\n=== ReduceLROnPlateau Demo ===")
# فرض کنید validation loss به صورت decreasing random داریم
import random
val_losses = [random.uniform(0.5, 1.5) - 0.05*step for step in range(steps)]
for step, val_loss in enumerate(val_losses):
    optimizer_plateau.step()
    scheduler_plateau.step(val_loss)  # توجه: باید loss validation بدیم
    print(f"Step {step + 1}: LR = {optimizer_plateau.param_groups[0]['lr']:.5f}, val_loss={val_loss:.4f}")
