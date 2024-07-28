import torch

import torch_hawk
import torch_hawk._torch_hawk

n = 64
k = 64

x = torch.ones(n, k, dtype=torch.float16).cuda()
w = torch.ones(k, 4, dtype=torch.float16).cuda()
s = torch.ones(n, dtype=torch.float16).cuda()
o = torch.zeros(n, k, dtype=torch.float16).cuda()

print(x, w, s, o)

torch_hawk._torch_hawk.conv1d_forward_cuda(x, w, s, o, n, k)

print(o.cpu())