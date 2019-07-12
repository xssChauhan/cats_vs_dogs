import torchvision

import torch.nn as nn
import torch

from dataset import train_dataloader
import matplotlib.pyplot as plt

model_map = {
    "mobilenet": torchvision.models.mobilenet_v2,
    "densenet": torchvision.models.densenet121
}


def get_model(model: str, pretrained: bool = True):
    """
    Get a torchvision model
    :param model:
    :param pretrained:
    :return:
    """
    return model_map.get(model)(pretrained=pretrained)


def train(model, epochs=3):

    model = get_model(model)

    num_ftrs = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[500, 1000, 1500],
        gamma=0.5
    )

    itr=1
    p_itr=200
    model.train()

    total_loss = 0
    loss_list = []
    acc_list = []

    model = model.to("cuda")

    for epoch in range(epochs):

        for samples, labels in train_dataloader:
            samples, labels = samples.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            scheduler.step()
            if itr%p_itr == 0:
                pred = torch.argmax(output, dim=1)
                correct = pred.eq(labels)
                acc = torch.mean(correct.float())
                print(
                    '[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch + 1, epochs, itr,
                                                                                                total_loss / p_itr,
                                                                                                acc))
                loss_list.append(total_loss / p_itr)
                acc_list.append(acc)
                total_loss = 0
            itr += 1

    plt.plot(loss_list, label="loss")
    plt.plot(acc_list, label="accuracy")
    plt.legend()
    plt.title("training loss and accuracy")
    plt.show()

    torch.save(model, "model/1.pt")


if __name__ == "__main__":
    train("mobilenet")