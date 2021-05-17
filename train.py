import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from data import PotsdamDataset, PotsdamPatchesDataset
from models import HAFNet

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rgb_dir = "data/Potsdam/2_Ortho_RGB"
    dsm_dir = "data/Potsdam/1_DSM"
    labels_dir = "data/Potsdam/Building_Labels"

    # training hyperparameters
    epochs = 1
    lr = 0.001
    batch_size = 15

    # model
    model = HAFNet.HAFNet().to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # datasets
    potsdam_dataset = PotsdamDataset(rgb_dir, dsm_dir, labels_dir, patch_stride=128)
    potsdam_dataloader = DataLoader(potsdam_dataset, batch_size=1, shuffle=True)

    # training and test set definition
    tr_length = int(0.7 * len(potsdam_dataset))
    ts_length = len(potsdam_dataset) - tr_length
    tr_potsdam_dataset, ts_potsdam_dataset = random_split(potsdam_dataset, (tr_length, ts_length))

    # Loop over images and extract patches
    tr_potsdam_dataloader = DataLoader(tr_potsdam_dataset, batch_size=1, shuffle=True)

    # epochs
    for epoch in range(epochs):

        # loop over images in the original dataset
        for (image_idx, patches_set) in enumerate(tr_potsdam_dataloader):

            potsdam_patches_dataset = PotsdamPatchesDataset(patches_set)
            potsdam_patches_loader = DataLoader(potsdam_patches_dataset, batch_size=batch_size, shuffle=True)

            image_time = time.time()

            # loop over patches of single image in original dataset
            for (patch_idx, patch) in enumerate(potsdam_patches_loader):

                (_, rgb_patch, dsm_patch, label_patch) = patch
                cuda_time = time.time()
                rgb_patch, dsm_patch, label_patch = rgb_patch.to(device), dsm_patch.to(device), label_patch.to(device)
                print('\n------------------------------')
                print('cuda time', time.time() - cuda_time)
                optimizer.zero_grad()
                forward_time = time.time()
                pred = model(rgb_patch, dsm_patch)
                # loss = criterion(pred, label_patch)
                # optimizer.step()
                # print(loss)
                print('forward time', time.time() - forward_time)
                print(image_idx, len(tr_potsdam_dataloader), patch_idx, len(potsdam_patches_loader), pred.shape)

                # print(image_idx, patch_idx, rgb_patch.shape, dsm_patch.shape, label_patch.shape)

            print('image time ', time.time() - image_time)
            break