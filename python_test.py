import numpy as np
import torch.nn as nn
import torch

ar = np.array([1, 0])
br = np.array([0.8, 0.2])
a = torch.Tensor(ar)
b = torch.Tensor(br)

loss = nn.CrossEntropyLoss()
loss_1 = nn.MSELoss()

print(loss(a,b))
print(loss_1(a,b))

