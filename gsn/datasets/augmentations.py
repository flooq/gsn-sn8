import albumentations as A


class GeometricTransform:
    def __init__(self, vertical_flip: bool, rotate_90: int):
        self.transform = A.Compose(
            [A.VerticalFlip(p=int(vertical_flip))] + [A.RandomRotate90(p=1)] * rotate_90
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


rotation_range = range(4)
flip_range = (True, False)
n_color_transforms = 5
spatial_augmentations = [GeometricTransform(vertical_flip=flip, rotate_90=rotate) for flip in flip_range for rotate in rotation_range]
color_augmentations = [ColorTransform() for _ in range(n_color_transforms)]