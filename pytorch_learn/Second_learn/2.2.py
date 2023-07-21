import torch

mean = torch.tensor([2, 3, 4])
mean = mean[:, None, None]

data = torch.ones((3, 3, 3))

result = data - mean

print(result)
