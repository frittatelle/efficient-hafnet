import os

import torch.cuda
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class PotsdamRGBDataset(Dataset):
    """Potsdam TOP RGB dataset"""

    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        image = transforms.ToTensor()(Image.open(img_path))
        label = transforms.ToTensor()(Image.open(label_path))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        item = {"image": image, "label": label}
        return item


training_data = PotsdamRGBDataset("data/Potsdam/2_Ortho_RGB", "data/Potsdam/Building_Labels")
train_loader = DataLoader(training_data, batch_size=2, shuffle=True)

# batch = next(iter(train_loader))
# fig, axs = plt.subplots(1,2)
# axs[0].imshow(batch['image'][0].squeeze().permute(1, 2, 0))
# axs[1].imshow(batch['label'][0].squeeze().permute(1, 2, 0))
# plt.show()