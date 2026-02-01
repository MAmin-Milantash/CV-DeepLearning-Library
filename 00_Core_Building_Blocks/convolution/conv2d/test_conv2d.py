import torch
from conv2d_from_scratch import Conv2DFromScratch
from conv2d_pytorch import Conv2DPyTorch

x = torch.randn(1, 3, 32, 32)

custom_conv = Conv2DFromScratch(3, 8, kernel_size=3, stride=1, padding=1)
torch_conv = Conv2DPyTorch(3, 8, kernel_size=3, stride=1, padding=1)

out_custom = custom_conv(x)
out_torch = torch_conv(x)

print("Custom Conv Output Shape:", out_custom.shape)
print("Torch Conv Output Shape:", out_torch.shape)
