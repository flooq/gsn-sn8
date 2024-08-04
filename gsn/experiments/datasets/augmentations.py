import albumentations as A

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
