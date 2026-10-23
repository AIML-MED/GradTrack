import clip
import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, in_dim=512, hidden=512, num_classes=100):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(in_dim, hidden), nn.LeakyReLU(0.1))
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.fc(self.pre(x))


def load_clip(device):
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode(model, images):
    return model.encode_image(images).float()
