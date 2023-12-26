import torch
import torch.nn as nn

criterion = nn.MSELoss()

predicted = torch.tensor([0,1], dtype=torch.float32)
reward = torch.tensor([-1,1], dtype=torch.float32)
predicted2 = torch.tensor([1,0], dtype=torch.float32)

print(criterion(predicted, reward).item())
print(criterion(predicted2, reward).item())