import os

import torch.cuda
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchio as tio

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
        image = transforms.ToTensor()(np.array(Image.open(img_path)))
        label = transforms.ToTensor()(np.array(Image.open(label_path)))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        item = {"image": image, "label": label}
        return item

class PotsdamDataset(Dataset):

    """Potsdam"""

    def __init__(self, rgb_dir, dsm_dir, labels_dir, transform=None, patch_size=128, patch_stride=128):
        # list rgb dir, list dsm dir, list label dir
        self.rgb_dir = rgb_dir
        self.dsm_dir = dsm_dir
        self.labels_dir = labels_dir
        self.rgb_fns = sorted(os.listdir(rgb_dir))
        self.dsm_fns = sorted(os.listdir(dsm_dir))
        self.labels_fns = sorted(os.listdir(labels_dir))
        self.transform = transform
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def __len__(self):
        return len(self.rgb_fns)

    def __getitem__(self, idx):
        # filenames
        rgb_fn = os.path.join(self.rgb_dir, self.rgb_fns[idx])
        dsm_fn = os.path.join(self.dsm_dir, self.dsm_fns[idx])
        labels_fn = os.path.join(self.labels_dir, self.labels_fns[idx])
        # read images
        rgb = transforms.ToTensor()(np.array(Image.open(rgb_fn)))
        dsm = transforms.ToTensor()(np.array(Image.open(dsm_fn)))
        labels = transforms.ToTensor()(np.array(Image.open(labels_fn)))
        # extract patches
        rgb_patches = self.extract_patches(rgb, self.patch_size, self.patch_stride)
        dsm_patches = self.extract_patches(dsm, self.patch_size, self.patch_stride, channels=1)
        labels_patches = self.extract_patches(labels, self.patch_size, self.patch_stride)
        # TODO: apply augmentation
        patches = {
            "id": rgb_fn,
            "rgb_patches": rgb_patches,
            "dsm_patches": dsm_patches,
            "labels_patches": labels_patches
        }
        return patches

    @staticmethod
    def extract_patches(x, size=128, stride=64, channels=3):
        patches = x.unfold(0, channels, channels).unfold(1, size, stride).unfold(2, size, stride)
        n_patches = pow(patches.shape[1], 2)
        patches = torch.reshape(patches, (n_patches, channels, size, size))
        return patches

if __name__ == '__main__':
    import time
    start_time = time.time()
    training_data = PotsdamDataset("data/Potsdam/2_Ortho_RGB", "data/Potsdam/1_DSM", "data/Potsdam/Building_Labels")
    train_loader = DataLoader(training_data, batch_size=1, shuffle=True)
    x = next(iter(train_loader))
    print(x['id'], x['rgb_patches'].shape, x['dsm_patches'].shape, x['labels_patches'].shape)
    print("--- %s seconds ---" % (time.time() - start_time))
    # training_data = PotsdamRGBDataset("data/Potsdam/2_Ortho_RGB", "data/Potsdam/Building_Labels")
    # train_loader = DataLoader(training_data, batch_size=2, shuffle=True)
    #
    # batch = next(iter(train_loader))
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(x['rgb_patches'][0][0].permute(1, 2, 0))
    axs[1].imshow(x['labels_patches'][0][0].permute(1, 2, 0))
    axs[2].imshow(x['dsm_patches'][0][0].permute(1, 2, 0))
    plt.show()
