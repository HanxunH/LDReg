from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class SimCLRTrainTransform(object):
    def __init__(self, cj_strength=1.0):
        color_jitter = transforms.ColorJitter(
            cj_strength * 0.8, cj_strength * 0.8, cj_strength * 0.8, cj_strength * 0.2
        )
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [color_jitter],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.5),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2