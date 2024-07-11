from typing import Optional, List

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder


class FloodSimpleModel(nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 3,
            device: str = None
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

        # self.flood_buildings = SegmentationHead(
        #     in_channels=decoder_channels[-1],
        #     out_channels=2, activation=None, kernel_size=3)

        # self.flood_roads = SegmentationHead(
        #     in_channels=decoder_channels[-1],
        #     out_channels=2, activation=None, kernel_size=3)

        self.penultimate_conv = nn.Conv2d(decoder_channels[-1]*2, 64, kernel_size=3, padding=1)
        self.outc1 = nn.Conv2d(64, 5, kernel_size=1)

        self.device = device
        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        # init.initialize_head(self.flood_buildings)
        # init.initialize_head(self.flood_roads)

    def forward(self, x1, x2):
        features_x1 = self.encoder(x1)
        features_x2 = self.encoder(x2)
        decoder_output_x1 = self.decoder(*features_x1)
        decoder_output_x2 = self.decoder(*features_x2)

        features = torch.cat([decoder_output_x1, decoder_output_x2], dim=1)

        x = self.penultimate_conv(features)
        x = self.outc1(x)

        return x

    @torch.no_grad()
    def predict(self, x1, x2):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x1, x2)
        return x


if __name__ == "__main__":
    model = FloodSimpleModel(encoder_name="resnet50")
    print(model)

    model.eval()
    preimg = torch.ones([1, 3, 1024, 1024], dtype=torch.float32)
    postimg = torch.ones([1, 3, 1024, 1024], dtype=torch.float32)
    x = model(preimg, postimg)
    print('flood_buildings', x.size())
