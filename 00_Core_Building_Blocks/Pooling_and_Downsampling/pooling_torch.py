import torch
import torch.nn as nn

x = torch.randn(1, 1, 8, 8)

max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

out_max = max_pool(x)
out_avg = avg_pool(x)

print("Input shape:", x.shape)
print("MaxPool output:", out_max.shape)
print("AvgPool output:", out_avg.shape)
