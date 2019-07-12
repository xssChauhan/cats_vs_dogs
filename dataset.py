import os

from torchvision import transforms

from PIL import Image

from torch.utils.data import Dataset, DataLoader, ConcatDataset


train_dir = "data/train/"
test_dir = "data/test1"


class CatDogDataset(Dataset):

    def __init__(self, file_list, dir, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir
        self.transform = transform
        self.mode = mode

        if self.mode == "train":
            if "dog" in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        img = Image.open(
            os.path.join(
                self.dir, self.file_list[idx]
            )
        )

        if self.transform:
            img = self.transform(img)

        img = img.numpy()

        if self.mode == "train":
            return img.astype("float32"), self.label
        else:
            return img.astype('float32'), self.file_list[idx]


data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset =  CatDogDataset(
        os.listdir(train_dir), train_dir, transform=data_transform
    )

test_dataset = CatDogDataset(
    os.listdir(test_dir), test_dir, transform=test_transform
)


train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4
)


test_dataloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4
)

