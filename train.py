import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from data import PotsdamDataset, PotsdamPatchesDataset
from models import HAFNet

import wandb


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        """
       I am assuming the model does not have sigmoid layer in the end. if that is the case, change torch.sigmoid(logits) to simply logits
        """
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


if __name__ == '__main__':

    wandb.init(project="icarus", reinit=True)

    plt.ion()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rgb_dir = "data/Potsdam/2_Ortho_RGB"
    dsm_dir = "data/Potsdam/1_DSM_normalisation"
    labels_dir = "data/Potsdam/Building_Labels"

    # training hyperparameters
    epochs = 1
    lr = 0.01
    batch_size = 10

    # model
    model = HAFNet.HAFNet(out_channel=1)
    model.cuda()

    wandb.watch(model, log_freq=10)

    rgb_parameters = [param for name, param in model.named_parameters() if 'RGB_E' in name]
    parameters = [param for name, param in model.named_parameters() if 'RGB_E' not in name]


    def weights_init(p):
        if isinstance(p, nn.Conv2d):
            nn.init.kaiming_normal_(p.weight)
            nn.init.zeros_(p.bias)
        if isinstance(p, nn.BatchNorm2d):
            nn.init.constant_(p.weight, 1)
            nn.init.constant_(p.bias, 0)
        return p


    parameters = [weights_init(param) for param in parameters]

    # loss and optimizer
    # pos_weight = torch.tensor(1.4)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.NLLLoss()
    rgb_optimizer = optim.SGD(rgb_parameters, lr=lr / 10, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.0005)

    milestones = [10, 15, 20]
    rgb_optim_scheduler = optim.lr_scheduler.MultiStepLR(rgb_optimizer, milestones=milestones, gamma=0.1)
    optim_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # transforms
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dsm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])
    labels_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])

    # datasets
    potsdam_dataset = PotsdamDataset(rgb_dir,
                                     dsm_dir,
                                     labels_dir,
                                     rgb_transform=rgb_transform,
                                     dsm_transform=dsm_transform,
                                     labels_transform=labels_transform,
                                     patch_size=128,
                                     patch_stride=64)
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

                rgb_id, dsm_id, labels_id, rgb_patch, dsm_patch, label_patch = patch['rgb_id'], patch['dsm_id'], \
                                                                               patch['labels_id'], patch['rgb'], \
                                                                               patch['dsm'], patch['label']
                cuda_time = time.time()
                rgb_patch, dsm_patch, label_patch = rgb_patch.cuda(), dsm_patch.cuda(), label_patch.cuda()
                print('\n------------------------------')
                print('cuda time', time.time() - cuda_time)
                rgb_optimizer.zero_grad()
                optimizer.zero_grad()
                forward_time = time.time()

                # rgb_patch, dsm_patch = rgb_patch.cpu(), dsm_patch.cpu()
                # plt.imshow(rgb_patch[0].permute(1,2,0))
                # plt.show()
                pred = model(rgb_patch, dsm_patch)
                print('forward time', time.time() - forward_time)
                loss = criterion(pred, label_patch)
                loss.backward()
                rgb_optimizer.step()
                optimizer.step()
                print('loss ', loss.item())
                if patch_idx % 10 == 0:
                    wandb.log({
                        "loss": loss
                    })
                if patch_idx % 200 == 0:
                    pred = torch.sigmoid(pred)
                    pred = pred.cpu()
                    rgb_patch = rgb_patch.cpu()
                    unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    rgb_patch = unnorm(rgb_patch)
                    dsm_patch = dsm_patch.cpu()
                    label_patch = label_patch.cpu()
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                    ax1.imshow(rgb_patch[0].squeeze().permute(1, 2, 0))
                    ax2.imshow(dsm_patch[0].squeeze())
                    ax3.imshow(pred[0].detach().squeeze())
                    ax4.imshow(label_patch[0].squeeze())
                    plt.show()
                    plt.pause(0.001)
                    torch.save(model, 'models/hafnet.pt')
                print(image_idx, len(tr_potsdam_dataloader), patch_idx, len(potsdam_patches_loader), pred.shape)

                # print(image_idx, patch_idx, rgb_patch.shape, dsm_patch.shape, label_patch.shape)
            print('image time ', time.time() - image_time)
            i = i + 1
            if i == 13:
                break

    rgb_optim_scheduler.step()
    optim_scheduler.step()
