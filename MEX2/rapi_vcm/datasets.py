"""PyTorch datasets for the command and wake-word tasks (training only)."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import spec_augment, time_shift


def _rng(worker_id: int) -> np.random.Generator:
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    return np.random.default_rng(seed)


class CommandDataset(Dataset):
    """(features, intent, slot) triples with SpecAugment on the fly."""

    def __init__(self, feat_dir: str, split: int, augment: bool = True):
        self.X = np.load(os.path.join(feat_dir, "cmd_X.npy"), mmap_mode="r")
        self.intent = np.load(os.path.join(feat_dir, "cmd_intent.npy"))
        self.slot = np.load(os.path.join(feat_dir, "cmd_slot.npy"))
        self.split = np.load(os.path.join(feat_dir, "cmd_split.npy"))
        self.idx = np.where(self.split == split)[0]
        self.augment = augment

    def __len__(self) -> int:
        return self.idx.shape[0]

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        feats = np.array(self.X[j], dtype=np.float32)
        if self.augment:
            rng = np.random.default_rng(
                (torch.initial_seed() + i) % (2 ** 32))
            feats = spec_augment(feats, rng)
            feats = time_shift(feats, rng, max_shift=12)
        return (torch.from_numpy(feats).unsqueeze(0),
                int(self.intent[j]), int(self.slot[j]))


class KeywordDataset(Dataset):
    """(features, multi-hot keyword targets, gold intent, gold slot)."""

    def __init__(self, feat_dir: str, split: int, augment: bool = True):
        self.X = np.load(os.path.join(feat_dir, "cmd_X.npy"), mmap_mode="r")
        self.kw = np.load(os.path.join(feat_dir, "cmd_kw.npy"))
        self.intent = np.load(os.path.join(feat_dir, "cmd_intent.npy"))
        self.slot = np.load(os.path.join(feat_dir, "cmd_slot.npy"))
        self.split = np.load(os.path.join(feat_dir, "cmd_split.npy"))
        self.idx = np.where(self.split == split)[0]
        self.augment = augment

    def __len__(self) -> int:
        return self.idx.shape[0]

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        feats = np.array(self.X[j], dtype=np.float32)
        if self.augment:
            rng = np.random.default_rng(
                (torch.initial_seed() + i) % (2 ** 32))
            feats = spec_augment(feats, rng)
            feats = time_shift(feats, rng, max_shift=12)
        return (torch.from_numpy(feats).unsqueeze(0),
                torch.from_numpy(self.kw[j].astype(np.float32)),
                int(self.intent[j]), int(self.slot[j]))


class WakeDataset(Dataset):
    """Binary wake-word windows."""

    def __init__(self, feat_dir: str, split: int, augment: bool = True):
        self.X = np.load(os.path.join(feat_dir, "wake_X.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(feat_dir, "wake_y.npy"))
        self.split = np.load(os.path.join(feat_dir, "wake_split.npy"))
        self.idx = np.where(self.split == split)[0]
        self.augment = augment

    def __len__(self) -> int:
        return self.idx.shape[0]

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        feats = np.array(self.X[j], dtype=np.float32)
        if self.augment:
            rng = np.random.default_rng((torch.initial_seed() + i) % (2 ** 32))
            feats = spec_augment(feats, rng, freq_mask=6, time_mask=16,
                                 n_freq=1, n_time=2)
            feats = time_shift(feats, rng, max_shift=8)
        return torch.from_numpy(feats).unsqueeze(0), int(self.y[j])


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v: float, k: int = 1):
        self.sum += float(v) * k
        self.n += k

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)
