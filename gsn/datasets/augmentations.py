import albumentations as A
from itertools import combinations_with_replacement

class GeometricTransform:
    def __init__(self, vertical_flip: bool, horizontal_flip: bool, transpose: bool):
        self.transform = A.Compose(
            [
                A.VerticalFlip(p=int(vertical_flip)),
                A.HorizontalFlip(p=int(horizontal_flip)),
                A.Transpose(p=int(transpose))
            ]
        )

    def __call__(self, image):
        transformed = self.transform(image=image)
        return transformed["image"]


class ColorTransform:
    def __init__(self, brightness: float=0.2, contrast: float=0.2, saturation: float=30, hue: float=20):
        self.transform = A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=brightness, contrast_limit=contrast, p=1),
                A.HueSaturationValue(hue_shift_limit=hue, sat_shift_limit=saturation, val_shift_limit=saturation, p=1),
            ]
        )

    def __call__(self, image):
        transformed = self.transform(image=image)
        return transformed["image"]


# All combinations of the three flips generate the D_8 group
spatial_augmentations = [GeometricTransform(*c) for c in combinations_with_replacement((False, True), 3)]
# Color augmentations are randomized
n_color_transforms = 4
color_augmentations = [lambda x: x] + [ColorTransform() for _ in range(n_color_transforms)]