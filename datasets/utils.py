import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageNet, GTSRB, STL10, Food101, StanfordCars, DTD
from torchvision.datasets.folder import ImageFolder
from .birdsnap import Birdsnap
from .collate import SimCLRCollateFunction, DefaultCollateFunction, BYOLCollateFunction
from .transforms_helper import CenterCropAndResize
from .simclr_augmentation import SimCLRTrainTransform
from .vicreg_augmentation import VICRegTrainTransform

transform_options = {
    "None": {
        "train_transform": None,
        "test_transform": None
        },
    "ToTensor": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR10_MAE": {
        "train_transform": [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "test_transform": [transforms.ToTensor()]
    },
    "CIFAR10LinearProb": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ],
        "test_transform": [transforms.ToTensor()]
        },
    "CIFAR10": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ],
        "test_transform": [transforms.ToTensor()]
        },
    "CIFAR100": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(15),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]
        },
    "GTSRB": {
        "train_transform": [
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ],
        "test_transform": [
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]},
    "ImageNet": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ]},
    "ImageNetNorm": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ],
        "test_transform": [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]},
    "ImageNetMAE": {
        "train_transform": [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ],
        "test_transform": [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]},
    "ImageNetLinearProb": {
        "train_transform": [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),],
        "test_transform": [
            CenterCropAndResize(proportion=0.875, size=224),
            transforms.ToTensor()]
    },
    "ImageNetLinearProbMAE": {
        "train_transform": [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        "test_transform": [
            CenterCropAndResize(proportion=0.875, size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    },
    "ImageNetLinearNorm": {
        "train_transform": [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        "test_transform": [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    },
    "SimCLR":{
        "train_transform": [
            SimCLRTrainTransform()
        ],
        "test_transform": [],
    },
    "VICReg":{
        "train_transform": [
            VICRegTrainTransform()
        ],
        "test_transform": [],
    },
    "StanfordCars":{
        "train_transform": [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()]
    },
    "CIFARLinearProb": {
        "train_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()]
    },
     "CIFARLinearProbMAE": {
        "train_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ],
        "test_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    },
    "STL10":{
        "train_transform": [
            transforms.RandomResizedCrop(96, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()],
        "test_transform": [transforms.Resize((96, 96)),
                           transforms.ToTensor()]
    },
}

dataset_options = {
        "CIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
        "CIFAR10C": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, key=kwargs['key'], transform=transform),
        "CIFAR100": lambda path, transform, is_test, kwargs:
        CIFAR100(root=path, train=not is_test, download=True,
                 transform=transform),
        "GTSRB": lambda path, transform, is_test, kwargs:
        GTSRB(root=path, split='test' if is_test else 'train', download=True,
              transform=transform),
        "SVHN": lambda path, transform, is_test, kwargs:
        SVHN(root=path, split='test' if is_test else 'train', download=True,
             transform=transform),
        "MNIST": lambda path, transform, is_test, kwargs:
        MNIST(root=path, train=not is_test, download=True,
              transform=transform),
        "ImageNet": lambda path, transform, is_test, kwargs:
        ImageNet(root=path, split='val' if is_test else 'train',
                 transform=transform),
        "ImageFolder": lambda path, transform, is_test, kwargs:
        ImageFolder(root=os.path.join(path, 'train') if not is_test else
                    os.path.join(path, 'val'),
                    transform=transform),
        "STL10_unsupervised": lambda path, transform, is_test, kwargs:
        STL10(root=path, split='unlabeled' if not is_test else 'test', transform=transform, download=True),
        "STL10_supervised": lambda path, transform, is_test, kwargs:
        STL10(root=path, split='train' if not is_test else 'test', transform=transform, download=True,
              folds=kwargs["folds"]),
        "FOOD101": lambda path, transform, is_test, kwargs:
        Food101(root=path, split='train' if not is_test else 'test', transform=transform, download=True),
        "StanfordCars": lambda path, transform, is_test, kwargs:
        StanfordCars(root=path,split='train' if not is_test else 'test',transform=transform, download=True),
        "Birdsnap": lambda path, transform, is_test, kwargs:
        Birdsnap(root=path, split='test' if is_test else 'train',transform=transform, download=True),
        "DTD": lambda path, transform, is_test, kwargs:
        DTD(root=path, split='train' if not is_test else 'test',transform=transform, download=True),
}


collate_fn_options = {
    'None': lambda **kwargs:
        DefaultCollateFunction(**kwargs),
    'SimCLR': lambda **kwargs:
        SimCLRCollateFunction(**kwargs),
    'BYOL': lambda **kwargs:
        BYOLCollateFunction(**kwargs),
}


def get_classidx(dataset_type, dataset):
    if 'CIFAR100' in dataset_type:
        return [
            np.where(np.array(dataset.targets) == i)[0] for i in range(100)
        ]
    elif 'CIFAR10' in dataset_type:
        return [np.where(np.array(dataset.targets) == i)[0] for i in range(10)]
    elif 'SVHN' in dataset_type:
        return [np.where(np.array(dataset.labels) == i)[0] for i in range(10)]
    else:
        error_msg = 'dataset_type %s not supported' % dataset_type
        raise(error_msg)
