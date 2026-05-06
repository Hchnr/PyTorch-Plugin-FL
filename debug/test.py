import torch_fl
import torch

device = "flagos:0"
B, M, K, N = 4, 128, 256, 64

torch.manual_seed(0)
a = torch.randn(B, M, K, device=device, dtype=torch.float32)
b = torch.randn(B, K, N, device=device, dtype=torch.float32)
out = torch.bmm(a, b)

torch.manual_seed(1)
a = torch.randn(4, 64, 128, device=device, dtype=torch.float32)
b = torch.randn(4, 128, 32, device=device, dtype=torch.float32)
out = torch.empty(4, 64, 32, device=device, dtype=torch.float32)
ret = torch.bmm(a, b, out=out)

torch.manual_seed(3)
a = torch.randn(4, 64, 128, device=device, dtype=torch.float16)
b = torch.randn(4, 128, 32, device=device, dtype=torch.float16)
out = torch.bmm(a, b)
