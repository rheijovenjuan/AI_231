"""Model architectures.

All networks share the same depthwise-separable convolutional backbone:

  * KeywordModel - multi-label head: one sigmoid per keyword (40 keywords),
                   the queries are classified by spotting which keywords are
                   present. The 19-class command + slot are recovered from
                   the keyword set by `rapi_vcm.keywords.keywords_to_intent`.
  * CommandModel  - the earlier multi-task baseline: 19-way intent logits +
                   slot-value logits (kept for comparison).
  * WakeModel     - single logit for the "Hey Rapi" wake word.

Everything is sized for a Raspberry Pi 4 (4 GB): < 130 K parameters, i.e.
well under 1 MB in float32.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .keywords import N_KEYWORDS
from .labels import INTENTS, SLOT_CLASSES


def _gn(ch: int) -> nn.GroupNorm:
    """GroupNorm with 8 groups - stable for small batches."""
    return nn.GroupNorm(min(8, ch), ch)


class DSConv(nn.Module):
    """Depthwise-separable convolution block (MobileNet v1 style)."""

    def __init__(self, cin: int, cout: int, stride=(1, 1)):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1,
                            groups=cin, bias=False)
        self.dw_bn = _gn(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.pw_bn = _gn(cout)
        self.act = nn.ReLU6(inplace=True)
        self.use_res = (stride == (1, 1) and cin == cout)

    def forward(self, x):
        y = self.act(self.dw_bn(self.dw(x)))
        y = self.act(self.pw_bn(self.pw(y)))
        if self.use_res:
            y = y + x
        return y


class Backbone(nn.Module):
    """(B, 1, mel, frames) -> (B, out_dim) pooled descriptor."""

    def __init__(self, widths, strides, in_ch=1):
        super().__init__()
        layers = [nn.Conv2d(in_ch, widths[0], 3, padding=1, bias=False),
                  _gn(widths[0]), nn.ReLU6(inplace=True)]
        for i in range(len(widths) - 1):
            layers.append(DSConv(widths[i], widths[i + 1], strides[i]))
        # final block keeps the same width, only trims time
        layers.append(DSConv(widths[-1], widths[-1], strides[-1]))
        self.net = nn.Sequential(*layers)
        self.out_dim = widths[-1]

    def forward(self, x):
        y = self.net(x)
        return y.mean(dim=(2, 3))          # global average pool


class CommandModel(nn.Module):
    """Wake-independent command classifier with a slot-value head."""

    NAME = "command"

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.backbone = Backbone(
            widths=[64, 96, 128, 160, 192],
            strides=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 1)],
        )
        d = self.backbone.out_dim
        self.drop = nn.Dropout(dropout)
        self.intent_head = nn.Linear(d, len(INTENTS))
        self.slot_head = nn.Linear(d, len(SLOT_CLASSES))
        self.num_classes = len(INTENTS)

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return self.intent_head(f), self.slot_head(f)


class KeywordModel(nn.Module):
    """Multi-label keyword spotter over the 40-keyword vocabulary.

    Same backbone as CommandModel, but one independent sigmoid per keyword
    instead of a softmax: several keywords can be active in one query
    ("play next song" -> play + music + next).
    """

    NAME = "keyword"

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.backbone = Backbone(
            widths=[64, 96, 128, 160, 192],
            strides=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 1)],
        )
        d = self.backbone.out_dim
        self.drop = nn.Dropout(dropout)
        self.keyword_head = nn.Linear(d, N_KEYWORDS)
        self.num_classes = N_KEYWORDS

    def forward(self, x):
        return self.keyword_head(self.drop(self.backbone(x)))


class WakeModel(nn.Module):
    """Binary 'Hey Rapi' detector over a 1.2 s sliding window."""

    NAME = "wake"
    NUM_CLASSES = 2

    def __init__(self, dropout: float = 0.15):
        super().__init__()
        self.backbone = Backbone(
            widths=[32, 48, 64, 96],
            strides=[(2, 2), (2, 2), (2, 2), (2, 1)],
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x):
        return self.head(self.drop(self.backbone(x))).squeeze(-1)


def build_command() -> CommandModel:
    return CommandModel()


def build_keyword() -> KeywordModel:
    return KeywordModel()


def build_wake() -> WakeModel:
    return WakeModel()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, m, shape in (("command", build_command(), (2, 1, 40, 250)),
                           ("wake", build_wake(), (2, 1, 40, 120))):
        x = torch.zeros(shape)
        out = m(x)
        outs = out if isinstance(out, tuple) else (out,)
        print(f"{name:8s} params={count_parameters(m):7d} "
              f"input={shape} -> {[tuple(o.shape) for o in outs]}")
