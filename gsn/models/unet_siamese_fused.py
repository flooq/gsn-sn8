from typing import Optional, List

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import ClassificationHead
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.manet.decoder import MAnetDecoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder


class UnetSiameseFused(nn.Module):
    def __init__(
            self,
            distance_transform=None,
            flood_classification=None,
            attention=None,
            encoder_name: str = "resnet50",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 3,
            fuse='cat',
            unet_plus_plus: bool = False,
            global_context: bool = False
    ):
        super().__init__()

        if distance_transform is None:
            distance_transform = {'enabled': False}

        if flood_classification is None:
            flood_classification = {'enabled': False}

        if attention is None:
            attention = {'enabled': False}

        self.unet_plus_plus = unet_plus_plus
        self.global_context = global_context

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if attention['enabled']:
            self.decoder = MAnetDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                pab_channels=attention.pab_channels
            )
        elif unet_plus_plus:
            self.decoder = UnetPlusPlusDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=None
            )
        else:
            self.decoder = UnetDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=None
            )
        if encoder_name in 'resnet34':
            ch = [3, 64, 64, 128, 256, 512]
        if encoder_name in  ['se_resnet50', 'resnet50', 'se_resnext50_32x4d']:
            ch = [3, 64, 256, 512, 1024, 2048]
        if encoder_name == 'timm-efficientnet-b0' or encoder_name == 'timm-efficientnet-b1':
            ch = [3, 32, 24, 40, 112, 320]
        if encoder_name == 'timm-efficientnet-b2':
            ch = [3, 32, 24, 48, 120, 352]
        if encoder_name == 'timm-efficientnet-b3':
            ch = [3, 40, 32, 48, 136, 384]
        if encoder_name == 'timm-efficientnet-b5':
            ch = [3, 48, 40, 64, 176, 512]
        if encoder_name == 'timm-efficientnet-b6':
            ch = [3, 56, 40, 72, 200, 576]

        self.fuse = fuse

        if self.fuse == 'cat':
            t = 2
        elif self.fuse == 'cat_add':
            t = 2
        elif self.fuse == 'add':
            t = 1

        # Pass the encoder outputs after concatenation through a Grouped Convolution layer
        self.projs_0 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[0] * t, out_channels=ch[0], kernel_size=(1, 1), groups=ch[0])])
        self.projs_1 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[1] * t, out_channels=ch[1], kernel_size=(1, 1), groups=ch[1])])
        self.projs_2 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[2] * t, out_channels=ch[2], kernel_size=(1, 1), groups=ch[2])])
        self.projs_3 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[3] * t, out_channels=ch[3], kernel_size=(1, 1), groups=ch[3])])
        self.projs_4 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[4] * t, out_channels=ch[4], kernel_size=(1, 1), groups=ch[4])])
        self.projs_5 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[5] * t, out_channels=ch[5], kernel_size=(1, 1), groups=ch[5])])
        self.projs = {0: self.projs_0, 1: self.projs_1, 2: self.projs_2, 3: self.projs_3,
                      4: self.projs_4, 5: self.projs_5}
        # the segmentation head
        num_classes = 5
        if distance_transform['enabled']:
            num_classes += 4

        segmentation_head_in_channels = decoder_channels[-1]
        if self.global_context:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.context_conv = nn.Conv2d(self.encoder.out_channels[-1], 32, kernel_size=1)
            segmentation_head_in_channels = segmentation_head_in_channels + 32

        self.segmentation_head = SegmentationHead(
            in_channels=segmentation_head_in_channels,
            out_channels=num_classes, activation=None, kernel_size=3)

        if flood_classification['enabled']:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1],  classes=1)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)
        for i in range(6):
            init.initialize_decoder(self.projs[i])

    def forward(self, x1, x2):
        enc_1 = self.encoder(x1)
        enc_2 = self.encoder(x2)
        final_features = []
        for i in range(0, len(enc_1)):
            if self.fuse == 'cat':
                enc_fusion = torch.cat([enc_1[i], enc_2[i]], dim=1)
                final_features.append(self.projs[i][0](enc_fusion))
            elif self.fuse == 'add':
                enc_fusion = enc_1[i] + enc_2[i]
                final_features.append(enc_fusion)

            elif self.fuse == 'cat_add':
                enc_fusion_1 = torch.cat([enc_1[i], enc_2[i]], dim=1)
                enc_fusion_2 = enc_1[i] + enc_2[i]
                enc_fusion = self.projs[i][0](enc_fusion_1) + enc_fusion_2
                final_features.append(enc_fusion)

        decoder_output = self.decoder(*final_features)

        if self.global_context:
            global_context = self.global_avg_pool(final_features[-1])
            global_context = self.context_conv(global_context)

            _, _, h, w = decoder_output.shape
            global_context_expanded = global_context.expand(-1, -1, h, w)
            decoder_output = torch.cat((decoder_output, global_context_expanded), dim=1)

        out_segmentation = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            out_classification = self.classification_head(final_features[-1])
            return out_segmentation, out_classification
        else:
            return out_segmentation, None


    def predict(self, x1, x2):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x1, x2)
        return x


if __name__ == "__main__":
    model = UnetSiameseFused(encoder_name="resnet50", distance_transform={'enabled': True}, global_context=True)
    print(model)

    model.eval()
    preimg = torch.ones([2, 3, 1280, 1280], dtype=torch.float32)
    postimg = torch.ones([2, 3, 1280, 1280], dtype=torch.float32)
    floods, _ = model(preimg, postimg)
    print('floods', floods.size())
