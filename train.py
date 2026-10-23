import argparse
import os

import torch
import torch.nn.functional as F

from data import make_loader
from model import Head, encode, load_clip
from utils import set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise_type", default="symmetric", choices=["symmetric", "asymmetric", "instance"])
    p.add_argument("--noise_rate", type=float, default=0.6)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--ckpt_root", default="./checkpoints")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = load_clip(device)
    loader = make_loader(args.data_root, args.noise_type, args.noise_rate, shuffle=True,
                         batch_size=args.batch_size)
    head = Head().to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)

    ckpt_dir = os.path.join(args.ckpt_root, f"cifar100_{args.noise_type}_{args.noise_rate}")
    os.makedirs(ckpt_dir, exist_ok=True)

    n = len(loader.dataset)
    for epoch in range(args.epochs):
        head.train()
        total_loss, correct = 0.0, 0
        for images, labels, _, clean in loader:
            labels = labels.to(device)
            logits = head(encode(backbone, images.to(device)))
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1).cpu() == clean).sum().item()
        print(f"epoch {epoch:02d}  loss {total_loss / n:.4f}  clean_acc {correct / n:.4f}", flush=True)
        torch.save(head.state_dict(), os.path.join(ckpt_dir, f"head_{epoch}.pth"))


if __name__ == "__main__":
    main()
