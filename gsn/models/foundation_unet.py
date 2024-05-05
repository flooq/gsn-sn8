from typing import Optional, List

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder


class FoundationUnet(nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 3,
            out_channels: List[int] = (1, 8)
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None

        )

        self.buildings_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=out_channels[0], activation=None, kernel_size=3)

        self.road_speeds_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=out_channels[1], activation=None, kernel_size=3)

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.buildings_head)
        init.initialize_head(self.road_speeds_head)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        buildings = self.buildings_head(decoder_output)
        road_speeds = self.road_speeds_head(decoder_output)

        return buildings, road_speeds


    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()

        buildings, road_speeds = self.forward(x)

        return buildings, road_speeds


if __name__ == "__main__":
    model = FoundationUnet(encoder_name="resnet34")
    print(model)

    model.eval()
    preimg = torch.ones([1, 3, 1024, 1024], dtype=torch.float32)
    buildings_pred, road_speeds_pred = model(preimg)
    print('buildings', buildings_pred.size())
    print('road_speeds', road_speeds_pred.size())
