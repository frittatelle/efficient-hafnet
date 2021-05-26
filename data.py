import os

import torch.cuda
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

class PotsdamDataset(Dataset):
    """Potsdam"""

    def __init__(self, rgb_dir, dsm_dir, labels_dir, rgb_transform=None, dsm_transform=None, labels_transform=None, patch_size=128, patch_stride=64):
        # list rgb dir, list dsm dir, list label dir
        self.rgb_dir = rgb_dir
        self.dsm_dir = dsm_dir
        self.labels_dir = labels_dir
        self.rgb_fns = sorted(os.listdir(rgb_dir))
        self.dsm_fns = sorted(os.listdir(dsm_dir))
        self.labels_fns = sorted(os.listdir(labels_dir))
        self.rgb_transform = rgb_transform
        self.dsm_transform = dsm_transform
        self.labels_transform = labels_transform
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
        rgb = np.array(Image.open(rgb_fn))
        dsm = np.array(Image.open(dsm_fn))
        labels = np.array(Image.open(labels_fn))
        if self.rgb_transform:
            rgb = self.rgb_transform(rgb)
        if self.dsm_transform:
            dsm = self.dsm_transform(dsm)
        if self.labels_transform:
            labels = self.labels_transform(labels)
        # extract patches
        rgb_patches = self.extract_patches(rgb, self.patch_size, self.patch_stride)
        dsm_patches = self.extract_patches(dsm, self.patch_size, self.patch_stride, channels=1)
        labels_patches = self.extract_patches(labels, self.patch_size, self.patch_stride, channels=1)
        patches = {
            "id": rgb_fn,
            "rgb_patches": rgb_patches,
            "dsm_patches": dsm_patches,
            "labels_patches": labels_patches
        }

        patches = self.filter_patches(patches)
        return patches

    @staticmethod
    def extract_patches(x, size=128, stride=64, channels=3):
        patches = x.unfold(0, channels, channels).unfold(1, size, stride).unfold(2, size, stride)
        n_patches = pow(patches.shape[1], 2)
        patches = torch.reshape(patches, (n_patches, channels, size, size))
        return patches

    @staticmethod
    def filter_patches(patches_set):
        patches = patches_set
        patches_idx = []
        for (patch_idx, patch) in enumerate(patches_set['labels_patches']):
            tot_nonzero_pixels = torch.count_nonzero(patch)
            tot_pixels = torch.numel(patch)
            if tot_nonzero_pixels/tot_pixels > 0.01:
                patches_idx.append(patch_idx)
        # return patches where nonzero pixels > 0.01
        patches['rgb_patches'] = torch.index_select(patches['rgb_patches'], 0, index=torch.tensor(patches_idx))
        patches['dsm_patches'] = torch.index_select(patches['dsm_patches'], 0, index=torch.tensor(patches_idx))
        patches['labels_patches'] = torch.index_select(patches['labels_patches'], 0, index=torch.tensor(patches_idx))
        return patches


class PotsdamPatchesDataset(Dataset):
    """Potsdam Patches dataset: takes patches as input
     and apply transformations and stuff"""

    def __init__(self, patches, device='cuda'):
        self.device = device
        self.patches = patches

    def __len__(self):
        return self.patches["rgb_patches"].shape[1]

    def __getitem__(self, idx):
        patch_id = self.patches["id"][0]
        rgb_patch = self.patches["rgb_patches"][0][idx]
        dsm_patch = self.patches["dsm_patches"][0][idx]
        label_patch = self.patches["labels_patches"][0][idx]
        patch = {
            "id": patch_id,
            "rgb": rgb_patch,
            "dsm": dsm_patch,
            "label": label_patch,
        }
        return patch


if __name__ == '__main__':
    import time

    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dsm_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(0, 1)
    ])
    labels_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])

    start_time = time.time()
    potsdam_dataset = PotsdamDataset("data/Potsdam/2_Ortho_RGB",
                                     "data/Potsdam/1_DSM_normalisation",
                                     "data/Potsdam/Building_Labels",
                                     rgb_transform=rgb_transform,
                                     dsm_transform=dsm_transform,
                                     labels_transform=labels_transform,
                                     patch_size=300,
                                     patch_stride=300)
    potsdam_loader = DataLoader(potsdam_dataset, batch_size=1)
    potsdam_patches = next(iter(potsdam_loader))
    print("--- %s seconds ---" % (time.time() - start_time))

    potsdam_patches_dataset = PotsdamPatchesDataset(potsdam_patches)
    potsdam_patches_loader = DataLoader(potsdam_patches_dataset, batch_size=1, shuffle=True)
    patch = next(iter(potsdam_patches_loader))
    (patch_id, rgb_patch, dsm_patch, label_patch) = (patch['id'], patch['rgb'], patch['dsm'], patch['label'])
    print(rgb_patch.shape, dsm_patch.shape, label_patch.shape)