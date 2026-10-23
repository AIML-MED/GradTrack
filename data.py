import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100

from utils import worker_init_fn

NUM_CLASSES = 100
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def flip(labels, P, random_state=0):
    rng = np.random.RandomState(random_state)
    noisy = labels.copy()
    for i, y in enumerate(labels):
        noisy[i] = np.flatnonzero(rng.multinomial(1, P[y], 1)[0])[0]
    return noisy


def symmetric_noise(labels, rate):
    P = np.full((NUM_CLASSES, NUM_CLASSES), rate / (NUM_CLASSES - 1))
    np.fill_diagonal(P, 1.0 - rate)
    return flip(labels, P)


def asymmetric_noise(labels, rate, group=5):
    block = (1.0 - rate) * np.eye(group)
    for i in range(group):
        block[i, (i + 1) % group] = rate
    P = np.eye(NUM_CLASSES)
    for s in range(0, NUM_CLASSES, group):
        P[s:s + group, s:s + group] = block
    return flip(labels, P)


def instance_noise(images, labels, rate):
    np.random.seed(0)
    q = np.random.normal(loc=rate, scale=0.1, size=1000000)
    q = q[(q > 0) & (q < 1)][:len(labels)]
    w = [np.random.normal(loc=0, scale=1, size=(32 * 32 * 3, NUM_CLASSES)) for _ in range(NUM_CLASSES)]
    noisy = labels.copy()
    for i, (x, y) in enumerate(zip(images, labels)):
        p = x.flatten() @ w[y]
        p[y] = -1000000
        p = q[i] * F.softmax(torch.tensor(p), dim=0).numpy()
        p[y] = 1 - q[i]
        noisy[i] = np.random.choice(NUM_CLASSES, p=p / p.sum())
    return noisy


def noisy_labels(root, images, labels, noise_type, noise_rate):
    path = os.path.join(root, f"cifar100_{noise_type}_{noise_rate}.npy")
    if os.path.exists(path):
        return np.load(path)
    if noise_type == "symmetric":
        noisy = symmetric_noise(labels, noise_rate)
    elif noise_type == "asymmetric":
        noisy = asymmetric_noise(labels, noise_rate)
    elif noise_type == "instance":
        noisy = instance_noise(images, labels, noise_rate)
    else:
        raise ValueError(noise_type)
    np.save(path, noisy)
    return noisy


class NoisyCIFAR100(Dataset):
    def __init__(self, root, noise_type, noise_rate, transform):
        base = CIFAR100(root, train=True, download=True)
        self.images = base.data
        self.clean = np.asarray(base.targets, dtype=np.int64)
        self.noisy = noisy_labels(root, self.images, self.clean, noise_type, noise_rate)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), self.noisy[i], i, self.clean[i]


def train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def make_loader(root, noise_type, noise_rate, shuffle, batch_size=256, num_workers=4):
    dataset = NoisyCIFAR100(root, noise_type, noise_rate, train_transform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      worker_init_fn=worker_init_fn)
