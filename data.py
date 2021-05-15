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


class PotsdamDataset(Dataset):

    """Potsdam"""

    def __init__(self, rgb_dir, dsm_dir, labels_dir, transform=None, patch_size=128, patch_stride=64):
        # list rgb dir, list dsm dir, list label dir
        self.rgb_fns = sorted(os.listdir(rgb_dir))
        self.dsm_fns = sorted(os.listdir(dsm_dir))
        self.labels_fns = sorted(os.listdir(labels_dir))
        self.transform = transform
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        # initialize patches tensors
        self.rgb_patches = torch.empty((0, 3, self.patch_size, self.patch_size)).cuda()
        self.dsm_patches = torch.empty((0, 2, self.patch_size, self.patch_size)).cuda()
        self.labels_patches = torch.empty((0, 3, self.patch_size, self.patch_size)).cuda()
        # extract patches from every image and append them to the patches tensors
        for rgb_fn, dsm_fn, label_fn in zip(self.rgb_fns, self.dsm_fns, self.labels_fns):
            rgb = transforms.ToTensor()(Image.open(os.path.join(rgb_dir, rgb_fn))).cuda()
            dsm = transforms.ToTensor()(Image.open(os.path.join(dsm_dir, dsm_fn)))
            labels = transforms.ToTensor()(Image.open(os.path.join(labels_dir, label_fn))).cuda()
            rgb_patches = self.extract_patches(rgb, self.patch_size, self.patch_stride)
            dsm_patches = self.extract_patches(dsm, self.patch_size, self.patch_stride, channels=1)
            labels_patches = self.extract_patches(labels, self.patch_size, self.patch_stride)
            self.rgb_patches = torch.cat((self.rgb_patches, rgb_patches), 0)
            self.dsm_patches = torch.cat((self.dsm_patches, dsm_patches), 0)
            self.labels_patches = torch.cat((self.labels_patches, labels_patches), 0)

    def __len__(self):
        return self.rgb_patches.shape[0]

    def __getitem__(self, idx):
        # get rgb and dsm image from the list
        # apply data augmentation
        # return sample = {"rgb","dsm","label"}
        pass

    @staticmethod
    def extract_patches(x, size=128, stride=64, channels=3):
        patches = x.unfold(0, channels, channels).unfold(1, size, stride).unfold(2, size, stride)
        n_patches = pow(patches.shape[1], 2)
        patches = torch.reshape(patches, (n_patches, channels, size, size))
        return patches

if __name__ == '__main__':
    dataset = PotsdamDataset("data/Potsdam/2_Ortho_RGB", "data/Potsdam/1_DSM", "data/Potsdam/Building_Labels")
    print(dataset)

# training_data = PotsdamRGBDataset("data/Potsdam/2_Ortho_RGB", "data/Potsdam/Building_Labels")
# train_loader = DataLoader(training_data, batch_size=2, shuffle=True)
#
# batch = next(iter(train_loader))
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(batch['image'][0].squeeze().permute(1, 2, 0))
# axs[1].imshow(batch['label'][0].squeeze().permute(1, 2, 0))
# plt.show()
