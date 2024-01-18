import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import PIL
from PIL import Image, ImageOps
import random
from .gaussian_blur import GaussianBlur
from .rotation import RandomRotate
import PIL
import numpy as np


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        

class DefaultCollateFunction(nn.Module):
    def __init__(self, **kwargs):
        super(DefaultCollateFunction, self).__init__()

    def forward(self, batch):
        return torch.stack([item[0] for item in batch]), torch.tensor([item[1] for item in batch])


class BaseCollateFunction(nn.Module):
    def __init__(self, transform):
        super(BaseCollateFunction, self).__init__()
        self.transform = transform

    def forward(self, batch):

        batch_size = len(batch)

        # list of transformed images
        transforms = [self.transform(batch[i % batch_size][0]).unsqueeze_(0)
                      for i in range(2 * batch_size)]
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])

        # tuple of transforms
        transforms = (
            torch.cat(transforms[:batch_size], 0),
            torch.cat(transforms[batch_size:], 0)
        )

        return transforms, labels
    

class TwoViewCollateFunction(nn.Module):
    def __init__(self, transform1, transform2):
        super(TwoViewCollateFunction, self).__init__()
        self.transform1 = transform1
        self.transform2 = transform2
        
    def forward(self, batch):

        batch_size = len(batch)

        # list of transformed images
        transforms1 = [self.transform1(batch[i][0]).unsqueeze_(0)
                       for i in range(batch_size)]
        
        transforms2 = [self.transform2(batch[i][0]).unsqueeze_(0)
                       for i in range(batch_size)]
        
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])

        # tuple of transforms
        transforms = (
            torch.cat(transforms1, 0),
            torch.cat(transforms2, 0)
        )

        return transforms, labels

class BYOLCollateFunction(TwoViewCollateFunction):
    def __init__(self,
                 input_size: int = 224,
                 cj_prob: float = 0.8,
                 cj_bright: float = 0.4,
                 cj_contrast: float = 0.4,
                 cj_sat: float = 0.2,
                 cj_hue: float = 0.1,
                 min_scale: float = 0.08,
                 random_gray_scale: float = 0.2,
                 hf_prob: float = 0.5,
                 **kwargs
                 ):
        color_jitter = T.ColorJitter(
            cj_bright, cj_contrast, cj_sat, cj_hue
        )
        transform1 = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),    
            GaussianBlur(p=1.0),
            T.ToTensor(),
        ]
        transform2 = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),    
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            T.ToTensor(),
        ]
        if 'normalize' in kwargs and kwargs['normalize']:
            transform1.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]))
            transform2.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]))
        transform1 = T.Compose(transform1)
        transform2 = T.Compose(transform2)
        super(BYOLCollateFunction, self).__init__(transform1, transform2)


class ImageCollateFunction(BaseCollateFunction):
    def __init__(self,
                 input_size: int = 64,
                 cj_prob: float = 0.8,
                 cj_bright: float = 0.7,
                 cj_contrast: float = 0.7,
                 cj_sat: float = 0.7,
                 cj_hue: float = 0.2,
                 min_scale: float = 0.15,
                 random_gray_scale: float = 0.2,
                 gaussian_blur: float = 0.5,
                 kernel_size: float = 0.1,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.5,
                 rr_prob: float = 0.0,
                 **kwargs
                 ):
        color_jitter = T.ColorJitter(
            cj_bright, cj_contrast, cj_sat, cj_hue
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),    
            GaussianBlur(p=gaussian_blur),
            T.ToTensor(),
        ]
        transform = T.Compose(transform)

        super(ImageCollateFunction, self).__init__(transform)


class SimCLRCollateFunction(ImageCollateFunction):
    def __init__(self,
                 input_size: int = 224,
                 cj_prob: float = 0.8,
                 cj_strength: float = 0.5,
                 min_scale: float = 0.08,
                 random_gray_scale: float = 0.2,
                 gaussian_blur: float = 0.5,
                 kernel_size: float = 0.1,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.5,
                 rr_prob: float = 0.0,
                 **kwargs):

        super(SimCLRCollateFunction, self).__init__(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_bright=cj_strength * 0.8,
            cj_contrast=cj_strength * 0.8,
            cj_sat=cj_strength * 0.8,
            cj_hue=cj_strength * 0.2,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
        )

