import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from data import PotsdamDataset, PotsdamPatchesDataset
from models import HAFNet

import wandb

if __name__ == '__main__':

    wandb.init(project="efficient-hafnet")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rgb_dir = "data/Potsdam/2_Ortho_RGB"
    dsm_dir = "data/Potsdam/1_DSM"
    labels_dir = "data/Potsdam/Building_Labels"

    # training hyperparameters
    epochs = 1
    lr = 0.001
    batch_size = 25

    # model
    model = HAFNet.HAFNet(out_channel=1)
    model.cuda()

    wandb.watch(model, log_freq=10)

    # loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # datasets
    potsdam_dataset = PotsdamDataset(rgb_dir, dsm_dir, labels_dir, patch_stride=64)
    potsdam_dataloader = DataLoader(potsdam_dataset, batch_size=1, shuffle=True)

    # training and test set definition
    tr_length = int(0.7 * len(potsdam_dataset))
    ts_length = len(potsdam_dataset) - tr_length
    tr_potsdam_dataset, ts_potsdam_dataset = random_split(potsdam_dataset, (tr_length, ts_length))

    # Loop over images and extract patches
    tr_potsdam_dataloader = DataLoader(tr_potsdam_dataset, batch_size=1, shuffle=True)

    i = 0

    # epochs
    for epoch in range(epochs):

        # loop over images in the original dataset
        for (image_idx, patches_set) in enumerate(tr_potsdam_dataloader):

            potsdam_patches_dataset = PotsdamPatchesDataset(patches_set, device=device)
            potsdam_patches_loader = DataLoader(potsdam_patches_dataset, batch_size=batch_size, shuffle=True)

            image_time = time.time()

            # loop over patches of single image in original dataset
            for (patch_idx, patch) in enumerate(potsdam_patches_loader):

                patch_id, rgb_patch, dsm_patch, label_patch = patch['id'], patch['rgb'], patch['dsm'], patch['label']
                cuda_time = time.time()
                rgb_patch, dsm_patch, label_patch = rgb_patch.cuda(), dsm_patch.cuda(), label_patch.cuda()
                print('\n------------------------------')
                print('cuda time', time.time() - cuda_time)
                optimizer.zero_grad()
                forward_time = time.time()
                pred = model(rgb_patch, dsm_patch)
                print('forward time', time.time() - forward_time)
                loss = criterion(pred, label_patch)
                backprop_time = time.time()
                loss.backward()
                print('backprop time', time.time() - backprop_time)
                optimizer.step()
                print('loss ', loss.item())
                if patch_idx % 10 == 0:
                    wandb.log({
                        'loss': loss
                    })
                print(image_idx, len(tr_potsdam_dataloader), patch_idx, len(potsdam_patches_loader), pred.shape)

                # print(image_idx, patch_idx, rgb_patch.shape, dsm_patch.shape, label_patch.shape)

            print('image time ', time.time() - image_time)
            i = i + 1
            if i == 3:
                break