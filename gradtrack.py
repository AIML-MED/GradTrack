import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from data import make_loader
from model import Head, encode, load_clip
from utils import set_seed


@torch.no_grad()
def last_layer_grads(backbone, head, loader, device):
    n = len(loader.dataset)
    num_classes, hidden = head.fc.weight.shape
    G = torch.zeros(n, num_classes * hidden + num_classes, dtype=torch.float16)
    norms = torch.zeros(n)
    labels = torch.zeros(n, dtype=torch.long)
    noisy = torch.zeros(n, dtype=torch.bool)
    for images, y, idx, clean in loader:
        h = head.pre(encode(backbone, images.to(device)))
        e = head.fc(h).softmax(1) - F.one_hot(y.to(device), num_classes).float()
        g = torch.cat([(e[:, :, None] * h[:, None, :]).flatten(1), e], 1).half()
        G[idx] = g.cpu()
        norms[idx] = g.float().norm(dim=1).cpu()
        labels[idx] = y
        noisy[idx] = y != clean
    return G, norms, labels, noisy


@torch.no_grad()
def misalignment(G, norms, labels, partition, k_anchor, energy, max_rank, device):
    n = len(G)
    tail = min(max(1, round(partition / 100 * n)), max(1, n // 2 - 1))
    nondiv = torch.zeros(n, dtype=torch.bool)
    nondiv[torch.argsort(norms)[:n - tail]] = True

    _, S, V = torch.svd_lowrank(G[nondiv].float().to(device), q=min(max_rank, int(nondiv.sum()), G.shape[1]),
                                niter=8)
    ratio = (S ** 2).cumsum(0) / (S ** 2).sum().clamp_min(1e-8)
    V = V[:, :int(torch.searchsorted(ratio, torch.tensor(energy, device=device))) + 1]

    mis = torch.zeros(n)
    for c in labels.unique().tolist():
        idx_c = torch.where(labels == c)[0]
        pool = torch.where((labels == c) & nondiv)[0]
        if len(pool) == 0:
            pool = idx_c
        top = pool[torch.topk(-norms[pool], k=min(k_anchor, len(pool))).indices]
        anchor = G[top].float().mean(0).to(device) @ V
        proj = G[idx_c].float().to(device) @ V
        mis[idx_c] = (1.0 - F.cosine_similarity(proj, anchor[None], dim=1)).cpu()
    return mis


def rank(x):
    return np.argsort(np.argsort(x, axis=0), axis=0) / (len(x) - 1)


def detect(mag, mis, warmup, evr, max_comp=40):
    Rm, Rs = rank(mag[:, warmup:]), rank(mis[:, warmup:])
    X = np.hstack([Rm, Rs])
    max_c = min(max_comp, X.shape[1] - 1, X.shape[0] - 1)
    cum = PCA(n_components=max_c, random_state=0).fit(X).explained_variance_ratio_.cumsum()
    k = max(2, min(int(np.searchsorted(cum, evr)) + 1, max_c))
    Z = PCA(n_components=k, random_state=0).fit_transform(X)
    gmm = GaussianMixture(n_components=2, covariance_type="full", reg_covar=1e-6, random_state=0, n_init=5).fit(Z)
    comp = gmm.predict(Z)
    noisy_comp = int(Rm[comp == 0].mean() < Rm[comp == 1].mean())
    return gmm.predict_proba(Z)[:, noisy_comp] >= 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise_type", default="symmetric", choices=["symmetric", "asymmetric", "instance"])
    p.add_argument("--noise_rate", type=float, default=0.6)
    p.add_argument("--start_epoch", type=int, default=2)
    p.add_argument("--end_epoch", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--k_anchor", type=int, default=100)
    p.add_argument("--evr", type=float, default=0.85)
    p.add_argument("--energy", type=float, default=0.95)
    p.add_argument("--partition", type=int, default=33)
    p.add_argument("--max_rank", type=int, default=512)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--ckpt_root", default="./checkpoints")
    p.add_argument("--out_dir", default="./results")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"cifar100_{args.noise_type}_{args.noise_rate}"

    backbone = load_clip(device)
    loader = make_loader(args.data_root, args.noise_type, args.noise_rate, shuffle=False)
    head = Head().to(device).eval()

    mags, miss = [], []
    for epoch in tqdm(range(args.start_epoch, args.end_epoch), desc=tag):
        state = torch.load(os.path.join(args.ckpt_root, tag, f"head_{epoch}.pth"), map_location=device)
        head.load_state_dict(state)
        G, norms, labels, noisy = last_layer_grads(backbone, head, loader, device)
        mis = misalignment(G, norms, labels, args.partition, args.k_anchor, args.energy, args.max_rank, device)
        mags.append(norms.numpy())
        miss.append(mis.numpy())
        del G
        torch.cuda.empty_cache()

    y_true = noisy.numpy().astype(int)
    pred = detect(np.stack(mags, 1), np.stack(miss, 1), args.warmup, args.evr)
    f1 = 100 * f1_score(y_true, pred)
    print(f"{tag}  F1 {f1:.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as f:
        json.dump({"noise_type": args.noise_type, "noise_rate": args.noise_rate, "f1": round(f1, 2)}, f)


if __name__ == "__main__":
    main()
