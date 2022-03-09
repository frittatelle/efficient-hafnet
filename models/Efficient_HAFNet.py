import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin
from segmentation_models_pytorch.encoders.efficientnet import EfficientNetEncoder, _get_pretrained_settings

import time

from torchsummaryX import summary
import torchinfo


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        if in_channels < r:
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, in_channels * r, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(in_channels * r, in_channels, bias=False),
                nn.Sigmoid()
            )
        else:
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, in_channels // r, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(in_channels // r, in_channels, bias=False),
                nn.Sigmoid()
            )

    def forward(self, rgb, dsm, cross_modal=None):
        # Single tensor for squeeze excitation ops
        if cross_modal is not None:
            x = torch.cat((rgb, dsm, cross_modal), dim=1)
        else:
            x = torch.cat((rgb, dsm), dim=1)
        batch, channels, _, _ = x.shape
        theta = self.squeeze(x).view(batch, channels)
        theta = self.excitation(theta).view(batch, channels, 1, 1)
        # Tensor is weighted with theta
        x = x * theta
        # Channel-wise summation
        if cross_modal is not None:
            rgb_features, dsm_features, cm_features = torch.chunk(x, 3, dim=1)
            x = rgb_features + dsm_features + cm_features
        else:
            rgb_features, dsm_features = torch.chunk(x, 2, dim=1)
            x = rgb_features + dsm_features
        return x


class EfficientHAFNetCMEncoder(EfficientNetEncoder, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):
        super(EfficientHAFNetCMEncoder, self).__init__(stage_idxs=stage_idxs,
                                                       out_channels=out_channels,
                                                       model_name=model_name,
                                                       depth=5)
        self.se_blocks = self.get_se_blocks_list(out_channels)

    def forward(self, rgb_features, dsm_features):
        stages = self.get_stages()

        block_number = 0.
        drop_connect_rate = self._global_params.drop_connect_rate
        cross_modal_features = []
        # Dummy feature
        cross_modal_features.append(torch.ones(1, 1, 1, 1))
        # First fusion has only rgb and dsm features
        x = self.se_blocks[0](rgb_features[1], dsm_features[1])
        cross_modal_features.append(x)

        for i in range(2, self._depth + 1):
            # i + 1 : Identity stage in rgb and dsm features is skipped
            for module in stages[i]:
                drop_connect = drop_connect_rate * block_number / len(self._blocks)
                block_number += 1.
                x = module(x, drop_connect)
            cross_modal_features.append(x)
            x = self.se_blocks[i - 1](rgb_features[i], dsm_features[i], cross_modal_features[-1])

        return cross_modal_features

    def load_state_dict(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, strict=False, **kwargs)

    @staticmethod
    def get_se_blocks_list(out_channels):
        se_blocks = []
        # Remove Identity stage channel,
        # (not used to choose the number of input channels in SE fusion block)
        out_channels = list(out_channels)
        out_channels.pop(0)
        out_channels = tuple(out_channels)
        for idx, c in enumerate(out_channels):
            if idx == 0:
                se_blocks.append(SEBlock(in_channels=c*2))
            else:
                se_blocks.append(SEBlock(in_channels=c*3))
        return nn.ModuleList(se_blocks)


class EfficientHAFNet(nn.Module):
    def __init__(self, encoder_name, encoder_weights):
        super(EfficientHAFNet, self).__init__()
        self.rgb_stream = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        self.dsm_stream = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=1,
        )
        self.cross_modal_stream = smp.Unet(
            encoder_name='efficienthafnet-cm-b0'
        )
        self.decision_level_fusion_block = SEBlock(in_channels=3)

    def forward(self, rgb, dsm):
        rgb_features = self.rgb_stream.encoder(rgb)
        rgb_decoder_out = self.rgb_stream.decoder(*rgb_features)
        rgb_pred = self.rgb_stream.segmentation_head(rgb_decoder_out)

        dsm_features = self.dsm_stream.encoder(dsm)
        dsm_decoder_out = self.dsm_stream.decoder(*dsm_features)
        dsm_pred = self.dsm_stream.segmentation_head(dsm_decoder_out)

        cross_modal_features = self.cross_modal_stream.encoder(rgb_features, dsm_features)
        cross_modal_decoder_out = self.cross_modal_stream.decoder(*cross_modal_features)
        cross_modal_pred = self.cross_modal_stream.segmentation_head(cross_modal_decoder_out)

        x = self.decision_level_fusion_block(rgb_pred, dsm_pred, cross_modal_pred)

        feats = {
            "rgb_features": rgb_features,
            "dsm_features": dsm_features,
            "cross_modal_features": cross_modal_features
        }
        preds = {
            "rgb_pred": rgb_pred,
            "dsm_pred": dsm_pred,
            "cross_modal_pred": cross_modal_pred
        }

        return x, feats, preds


smp.encoders.encoders["efficienthafnet-cm-b0"] = {
    "encoder": EfficientHAFNetCMEncoder,
    "pretrained_settings": _get_pretrained_settings("efficientnet-b0"),
    "params": {
        "out_channels": (3, 32, 24, 40, 112, 320),
        "stage_idxs": (3, 5, 9, 16),
        "model_name": "efficientnet-b0",
    },
}
smp.encoders.encoders["efficienthafnet-cm-b2"] = {
    "encoder": EfficientHAFNetCMEncoder,
    "pretrained_settings": _get_pretrained_settings("efficientnet-b2"),
    "params": {
        "out_channels": (3, 32, 24, 48, 120, 352),
        "stage_idxs": (5, 8, 16, 23),
        "model_name": "efficientnet-b2",
    },
}
smp.encoders.encoders["efficienthafnet-cm-b4"] = {
    "encoder": EfficientHAFNetCMEncoder,
    "pretrained_settings": _get_pretrained_settings("efficientnet-b4"),
    "params": {
        "out_channels": (3, 48, 32, 56, 160, 448),
        "stage_idxs": (6, 10, 22, 32),
        "model_name": "efficientnet-b4",
    },
}

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rgb = torch.randn(1, 3, 128, 128, device=device)
    dsm = torch.randn(1, 1, 128, 128, device=device)


    efficient_hafnet = EfficientHAFNet(encoder_name='efficientnet-b0', encoder_weights='imagenet')
    efficient_hafnet.to(device)
    efficient_hafnet.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


    inf_time = time.time()
    pred, feats, preds = efficient_hafnet(rgb, dsm)
    print(time.time() - inf_time)

    # summary(efficient_hafnet, rgb, dsm)
    # torchinfo.summary(model, input_size=(10, 3, 224, 224))


    from models.HAFNet import HAFNet
    hafnet = HAFNet(out_channel=1)
    hafnet.to(device)
    inf_time = time.time()
    p = hafnet(rgb, dsm)
    print(time.time() - inf_time)
    # summary(hafnet, rgb, dsm)
