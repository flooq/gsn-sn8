from typing import Optional, List

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import initialization as init, ClassificationHead
from segmentation_models_pytorch.decoders.manet.decoder import MAnetDecoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder


class UnetSiamese(nn.Module):
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
            device: str = "cuda"
    ):
        super().__init__()

        if distance_transform is None:
            distance_transform = {'enabled': False}

        if flood_classification is None:
            flood_classification = {'enabled': False}

        if attention is None:
            attention = {'enabled': False}

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if attention.enabled:
            self.decoder = MAnetDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                use_batchnorm=decoder_use_batchnorm,
                pab_channels=attention.pab_channels
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

        self.penultimate_conv = nn.Conv2d(decoder_channels[-1]*2, 64, kernel_size=3, padding=1)

        # the segmentation head
        num_classes = 5
        if distance_transform['enabled']:
            num_classes += 4

        self.outc1 = nn.Conv2d(64, num_classes, kernel_size=1)

        if flood_classification['enabled']:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1]*2, classes=1)
        else:
            self.classification_head = None

        self.device = device
        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x1, x2):
        features_x1 = self.encoder(x1)
        features_x2 = self.encoder(x2)
        decoder_output_x1 = self.decoder(*features_x1)
        decoder_output_x2 = self.decoder(*features_x2)

        features = torch.cat([decoder_output_x1, decoder_output_x2], dim=1)

        x = self.penultimate_conv(features)
        x = self.outc1(x)

        if self.classification_head is not None:
            combined_features = torch.cat([features_x1[-1], features_x2[-1]], dim=1)
            out_classification = self.classification_head(combined_features)
            return x, out_classification
        else:
            return x, None


    @torch.no_grad()
    def predict(self, x1, x2):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x1, x2)
        return x


if __name__ == "__main__":
    model = UnetSiamese(encoder_name="resnet50", distance_transform = {'enabled': True}, flood_classification = {'enabled': True})
    print(model)

    model.eval()
    preimg = torch.ones([2, 3, 1280, 1280], dtype=torch.float32)
    postimg = torch.ones([2, 3, 1280, 1280], dtype=torch.float32)
    x,_ = model(preimg, postimg)
    print('flood_buildings', x.size())
