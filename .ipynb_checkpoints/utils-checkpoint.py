import os

import numpy as np

import torch
import torch.nn as nn

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist

from pyro.contrib.examples.util import MNIST
import pyro.contrib.examples.util  # patches torchvision


# for loading and batching MNIST dataset
def make_loaders_mnist(
        batch_size=128,
        use_cuda=False
    ):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = MNIST(root=root, train=True, transform=trans,
                      download=download)
    test_set = MNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader



NORMALIZERS = {
    "cifar10": ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    "cifar100": ((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
    "svhn": ((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
}

def make_loaders_bnns(
        dataset: str,
        root: str,
        train_batch_size: int,
        test_batch_size: int,
        use_cuda: bool = False,
        mock_dataset: bool = False,
    ):
    if mock_dataset:
        # mock dataset:
        class MockData:
            def __init__(self, length, generator):
                self.generator = generator
                self.sampler = [0] * length
            def __iter__(self):
                for batch in self.generator:
                    yield batch
        B = train_batch_size
        BTest = test_batch_size
        train_batches = 5
        test_batches = 2
        num_classes = 10
        C, H, W = 3, 64, 64
        train_loader = MockData(B*train_batches, ((torch.randn(B,C,H,W), torch.randint(num_classes, (B,))) for _ in range(train_batches)))
        test_loader = MockData(BTest*test_batches, ((torch.randn(BTest,C,H,W), torch.randint(num_classes, (BTest,))) for _ in range(test_batches)))
        ood_loader = MockData(BTest*test_batches, ((torch.randn(BTest,C,H,W) * 2 + 3, torch.randint(num_classes, (BTest,))) for _ in range(test_batches)))
        return train_loader, test_loader, ood_loader
    
    train_img_transforms = [torchvision.transforms.RandomCrop(size=32, padding=4, padding_mode='reflect'),
                            torchvision.transforms.RandomHorizontalFlip()]
    test_img_transforms = []
    tensor_transforms = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(*NORMALIZERS[dataset])]

    dataset_fn = getattr(torchvision.datasets, dataset.upper())
    train_transform = torchvision.transforms.Compose(train_img_transforms + tensor_transforms)
    train_data = dataset_fn(root, train=True, transform=train_transform, download=True)
    train_loader = data.DataLoader(
        train_data, train_batch_size, pin_memory=use_cuda, num_workers=2 * int(use_cuda), shuffle=True)

    test_transform = torchvision.transforms.Compose(test_img_transforms + tensor_transforms)
    test_data = dataset_fn(root, train=False, transform=test_transform, download=True)
    test_loader = data.DataLoader(test_data, test_batch_size)

    ood_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(*NORMALIZERS["svhn"])])
    ood_data = torchvision.datasets.SVHN(root, split="test", transform=ood_transform, download=True)
    ood_loader = data.DataLoader(ood_data, test_batch_size)

    return train_loader, test_loader, ood_loader