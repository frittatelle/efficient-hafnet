import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from data import ISPRSDataset

import torchvision.transforms as transforms

from tqdm import tqdm

from models.Efficient_HAFNet import EfficientHAFNet

import wandb

from segmentation_models_pytorch.utils.functional import iou, f_score, accuracy

if __name__ == '__main__':

    rgb_dir = 'data/Isprs/RGB_patches'
    dsm_dir = 'data/Isprs/DSM_patches'
    labels_dir = 'data/Isprs/Labels_patches'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    ENCODER = 'efficientnet-b0'
    batch_size = 20
    lr = 0.001
    epochs = 10
    milestones = [2, 5]
    gamma = 0.1
    steps = batch_size
    tr_perc = 0.5 #to get an integer number of samples divisible by steps

    wandb.init(project='efficient-hafnet-vaihingen')

    model = EfficientHAFNet(
        encoder_name=ENCODER,
        encoder_weights='imagenet'
    )
    # model.load_state_dict(torch.load('models/efficienthafnet/whole-smoke-13.pth'))
    model.to(DEVICE)
    model.train()

    wandb.watch(model)

    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])

    ])
    dsm_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    label_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    isprs_dataset = ISPRSDataset(
        rgb_dir=rgb_dir,
        dsm_dir=dsm_dir,
        labels_dir=labels_dir,
        rgb_transform=rgb_transform,
        dsm_transform=dsm_transform,
        label_transform=label_transform,
    )

    # training, validation and test set definition
    tr_length = int(tr_perc * len(isprs_dataset))
    val_length = int((len(isprs_dataset) - tr_length) / 2)
    ts_length = int((len(isprs_dataset) - tr_length - val_length))
    tr_dataset, val_dataset, ts_dataset = random_split(isprs_dataset, (tr_length, val_length, ts_length))

    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    criterion = nn.BCEWithLogitsLoss()

    best_val_iou = 0.0

    tr_loss = 1.0
    tr_iou = 0.0
    tr_fscore = 0.0
    tr_accuracy = 0.0
    tr_precision = 0.0
    tr_recall = 0.0

    val_loss = 1.0
    val_iou = 0.0
    val_fscore = 0.0
    val_accuracy = 0.0
    val_precision = 0.0
    val_recall = 0.0

    for epoch in range(epochs + 1):
        print('--------------------------------------')
        print('epoch: ', epoch)

        print('training')
        model.train()
        tr_loop = tqdm(tr_dataloader)

        for batch_idx, batch in enumerate(tr_loop):
            rgb = batch['rgb'].to(DEVICE)
            dsm = batch['dsm'].to(DEVICE)
            label = batch['label'].to(DEVICE)

            pred, _, _ = model(rgb, dsm)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.sigmoid(pred)

            tr_loss += loss.item()
            tr_iou += iou(pred, label, threshold=0.5)
            tr_fscore += f_score(pred, label, threshold=0.5)
            tr_accuracy += accuracy(pred, label)

            if (batch_idx % steps == 0 and batch_idx != 0) or (batch_idx % steps == 0 and epoch > 0):

                tr_loss /= steps
                tr_iou /= steps
                tr_fscore /= steps
                tr_accuracy /= steps

                print('validation')
                model.eval()
                val_loop = tqdm(val_dataloader)

                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(val_loop):
                        rgb = val_batch['rgb'].to(DEVICE)
                        dsm = val_batch['dsm'].to(DEVICE)
                        label = val_batch['label'].to(DEVICE)

                        pred, _, _ = model(rgb, dsm)
                        loss = criterion(pred, label)

                        pred = torch.sigmoid(pred)
                        thr_pred = torch.round(pred)

                        val_loss += loss.item()
                        val_iou += iou(pred, label, threshold=0.1)
                        val_fscore += f_score(pred, label, threshold=0.1)
                        val_accuracy += accuracy(pred, label)

                    val_loss /= len(val_dataloader)
                    val_iou /= len(val_dataloader)
                    val_fscore /= len(val_dataloader)
                    val_accuracy /= len(val_dataloader)

                    wandb.log({
                        'training_loss': tr_loss,
                        'training_iou': tr_iou,
                        'training_fscore': tr_fscore,
                        'training_accuracy': tr_accuracy,
                        'validation_loss': val_loss,
                        'validation_iou': val_iou,
                        'validation_fscore': val_fscore,
                        'validation_accuracy': val_accuracy,
                        'rgb_input': wandb.Image(rgb[0:4]),
                        'dsm_input': wandb.Image(dsm[0:4]),
                        'label': wandb.Image(label[0:4]),
                        'prediction': wandb.Image(pred[0:4]),
                        'thresholded prediction': wandb.Image(thr_pred[0:4])
                    })

                    # logging
                    print('training loss: ', tr_loss)
                    print('validation loss: ', val_loss)
                    print('training iou', tr_iou)
                    print('validation iou', val_iou)
                    print('training fscore', tr_fscore)
                    print('validation fscore', val_fscore)
                    print('training accuracy', tr_accuracy)
                    print('validation accuracy', val_accuracy)

                    # save best model
                    if val_iou > best_val_iou:
                        # Update best iou
                        best_val_iou = val_iou
                        # Save weights for pytorch models
                        path = f'models/efficienthafnet/{wandb.run.name}.pth'
                        torch.save(model.state_dict(), path)

                    val_loss = 0.0
                    val_iou = 0.0
                    val_fscore = 0.0
                    val_accuracy = 0.0
                    tr_loss = 0.0
                    tr_iou = 0.0
                    tr_fscore = 0.0
                    tr_accuracy = 0.0

        lr_scheduler.step()
