import os

import torch.cuda
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

class PotsdamDataset(Dataset):

    """Potsdam"""

    def __init__(self, rgb_dir, dsm_dir, labels_dir, transform=None, patch_size=128, patch_stride=64):
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
        return self.rgb_fns.__len__()

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


class PotsdamPatchesDataset(Dataset):

    """Potsdam Patches dataset: takes patches as input
     and apply transformations and stuff"""

    def __init__(self, patches, transform=None, target_transform=None):
        self.patches = patches
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.patches["rgb_patches"].shape[1]

    def __getitem__(self, idx):
        patch_id = self.patches["id"][0]
        rgb_patch = self.patches["rgb_patches"][0][idx]
        dsm_patch = self.patches["dsm_patches"][0][idx]
        label_patch = self.patches["labels_patches"][0][idx]
        # TODO: apply augmentation
        if self.transform:
            rgb_patch = self.transform(rgb_patch)
            dsm_patch = self.transform(dsm_patch)
        if self.target_transform:
            label_patch = self.target_transform(label_patch)
        patch = {
            "id": patch_id,
            "rgb": rgb_patch,
            "dsm": dsm_patch,
            "label": label_patch,
        }
        return (patch_id, rgb_patch, dsm_patch, label_patch)


if __name__ == '__main__':
    import time
    start_time = time.time()
    potsdam_dataset = PotsdamDataset("data/Potsdam/2_Ortho_RGB", "data/Potsdam/1_DSM", "data/Potsdam/Building_Labels")
    potsdam_loader = DataLoader(potsdam_dataset, batch_size=1, shuffle=True)
    potsdam_patches = next(iter(potsdam_loader))

    potsdam_patches_dataset = PotsdamPatchesDataset(potsdam_patches)
    potsdam_patches_loader = DataLoader(potsdam_patches_dataset, batch_size=10, shuffle=True)
    patch = next(iter(potsdam_patches_loader))
    print(patch['rgb'].shape, patch['dsm'].shape, patch['label'].shape)
    print("--- %s seconds ---" % (time.time() - start_time))