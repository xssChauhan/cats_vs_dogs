from dataset import test_dataloader

import torch

model = torch.load("model/1.pt")

total = 0
correct = 0
model.eval()

for x, labels in test_dataloader:
    x, labels = x.cuda(), labels.cuda()
    with torch.no_grad():
        output = model(x)
        _, predicted = torch.max(output.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Correct ", correct )
print("Total ", total)
print("Acc : ", correct/total)

