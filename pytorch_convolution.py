import torch
import torch.nn as nn
import time

device = torch.device('mps')

image = torch.rand((3,28,28))

image = image.to(device)

cnn = nn.Conv2d(3, 32, 1, 1, padding=0)

cnn.to(device)

start = time.time()
print(cnn(image).shape)
end = time.time()

print(end - start)