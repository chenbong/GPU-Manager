import torch
import os
t = []
for i in range(torch.cuda.device_count()):
    t.append(torch.rand(1000, 1000).cuda(i))



while True:
    if not os.path.exists(__file__):
        break
    for i in range(torch.cuda.device_count()):
        torch.pow(t[i], 2)

