from typing import Optional, List

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder


class FloodModel(nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 3,
            concatenate_images: bool = True,
            add_images: bool = False,
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

        self.floods = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=5, activation=None, kernel_size=3)

        if not concatenate_images and not add_images:
            raise ValueError("Both concatenation and addition of images are disabled.")

        self.concatenate_images = concatenate_images
        self.add_images = add_images

        encoder_channels_dict = {
            'resnet34': [3, 64, 64, 128, 256, 512],
            'resnet50': [3, 64, 256, 512, 1024, 2048]
        }
        channels = encoder_channels_dict.get(encoder_name, None)
        if channels is None:
            raise ValueError("Encoder name '{}' not supported.".format(encoder_name))

        self.device = device
        multiplier = 2 if self.concatenate_images else 1
        self.blocks = []
        for channel in channels:
            conv2d = torch.nn.Conv2d(in_channels=channel * multiplier, out_channels=channel, kernel_size=(1, 1), groups=channel)
            if self.device is not None:
                conv2d.to(self.device)
            self.blocks.append(conv2d)

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.floods)
        for block in self.blocks:
            init.initialize_decoder(block)

    def forward(self, x1, x2):
        features_x1 = self.encoder(x1)
        features_x2 = self.encoder(x2)
        features = []
        for i, (f1, f2) in enumerate(zip(features_x1, features_x2)):
            con, cat = None, None
            if self.concatenate_images:
                con = torch.cat([f1, f2], dim=1)
                con = self.blocks[i](con)
            if self.add_images:
                cat = f1 + f2

            if con is not None and cat is not None:
                feature = con + cat
            elif con is not None:
                feature = con
            else:
                feature = cat
            features.append(feature)

        decoder_output = self.decoder(*features)
        flood_pred = self.floods(decoder_output)
        return flood_pred

    @torch.no_grad()
    def predict(self, x1, x2):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x1, x2)
        return x


if __name__ == "__main__":
    model = FloodModel(encoder_name="resnet50", concatenate_images=True)
    print(model)

    model.eval()
    preimg = torch.ones([1, 3, 1024, 1024], dtype=torch.float32)
    postimg = torch.ones([1, 3, 1024, 1024], dtype=torch.float32)
    floods = model(preimg, postimg)
    print('floods', floods.size())
