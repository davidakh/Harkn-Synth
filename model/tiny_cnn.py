"""TinyCNN: the first Electro model. A CQT spectrogram goes in, a guess at
which of the 120 neck positions was played comes out.

What a CNN does here
--------------------
A convolutional layer slides small learnable 3x3 "stencils" (kernels) over
the spectrogram. Each kernel computes a weighted sum of a 3x3 patch at every
location, so it lights up wherever its pattern appears: an onset edge, a
harmonic line, a fast-decaying smear... Stacking layers builds bigger
patterns out of smaller ones. The same kernel is reused at every location
("weight sharing"), which is why a CNN needs far fewer parameters than a
fully-connected network on the same image.

Each block: conv -> batchnorm -> ReLU -> pool
  * conv:      3x3 kernels, `out_channels` of them = that many pattern detectors.
  * batchnorm: rescales each channel to ~zero mean / unit spread over the
               batch, then applies a learned scale (gamma) and shift (beta).
               Keeps the numbers in a healthy range layer after layer, so
               training is faster and less sensitive to the learning rate.
  * ReLU:      max(0, x). The non-linearity. Without it, any stack of layers
               would collapse into one big linear function.
  * pool:      max over small windows, shrinking the image. Cheaper later
               layers, more tolerance to tiny shifts, and a bigger "view"
               (receptive field) for each later kernel.

The one design decision that matters most: pool over TIME, not FREQUENCY
------------------------------------------------------------------------
The spec says "global pooling". A standard image CNN averages its final
feature maps over *both* axes, because a cat is a cat wherever it is in the
photo. Here, *where* the energy sits on the frequency axis IS the pitch
(3 rows = 1 fret). Averaging over frequency would throw the pitch away. So:
  * time: pooled heavily, then globally averaged. A note is the same note
    whether its features are found at 0.2 s or 0.3 s.
  * frequency: pooled only 4x in total (252 -> 63 rows, each row = 4 bins
    = 1.33 semitones), then kept. The final linear layer looks at every
    (channel, frequency-row) value, so it knows both *what* patterns were
    found (timbre: brightness, decay) and *where* (pitch).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class TinyCNNConfig:
    n_bins: int = 252  # CQT rows (frequency)
    n_frames: int = 44  # CQT columns (time); 44 frames = first ~1.0 s
    input_seconds: float = 1.0  # what n_frames was derived from (for the record)
    n_classes: int = 120  # 6 strings x 20 frets (0-19)
    channels: tuple[int, ...] = (16, 32, 64, 32)
    # pooling (frequency, time) after each block: frequency 252->126->63, time 44->22->11->5->2
    pools: tuple[tuple[int, int], ...] = ((2, 2), (2, 2), (1, 2), (1, 2))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TinyCNNConfig":
        d = dict(d)
        d["channels"] = tuple(d["channels"])
        d["pools"] = tuple(tuple(p) for p in d["pools"])
        return cls(**d)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int]):
        super().__init__()
        # bias=False: batchnorm right after adds its own shift (beta), so a
        # conv bias would be redundant.
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(pool) if pool != (1, 1) else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class TinyCNN(nn.Module):
    def __init__(self, config: TinyCNNConfig):
        super().__init__()
        self.config = config
        min_frames = math.prod(time for _, time in config.pools)
        if config.n_frames < min_frames:
            raise ValueError(
                f"n_frames={config.n_frames} is too short: the time pooling shrinks it by "
                f"{min_frames}x, so at least {min_frames} frames are needed (use a longer input)"
            )
        blocks, in_ch = [], 1  # 1 input channel: the spectrogram is a grayscale image
        for out_ch, pool in zip(config.channels, config.pools):
            blocks.append(ConvBlock(in_ch, out_ch, pool))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Work out the feature-map size by pushing one dummy input through,
        # instead of doing the pooling arithmetic by hand.
        with torch.no_grad():
            dummy = self.blocks(torch.zeros(1, 1, config.n_bins, config.n_frames))
        self.n_channels_out, self.n_freq_out = dummy.shape[1], dummy.shape[2]

        # Final classifier: one weight per (feature value, class) pair.
        # Its 120 outputs are "logits": unnormalized scores, one per position.
        # Softmax turns them into probabilities summing to 1. We don't apply
        # it here because nn.CrossEntropyLoss does softmax + log internally,
        # in a numerically safer way.
        self.classifier = nn.Linear(self.n_channels_out * self.n_freq_out, config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, n_bins, n_frames) normalized CQT -> (batch, n_classes) logits."""
        h = self.blocks(x)  # (batch, channels, freq rows, time steps)
        h = h.mean(dim=3)  # global average pool over TIME only -> (batch, channels, freq rows)
        return self.classifier(h.flatten(1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_report(model: TinyCNN) -> str:
    """Human-readable breakdown of where the learnable numbers live.

    Parameters are the numbers training adjusts:
      * conv weights: out_ch x in_ch x 3 x 3. One 3x3 stencil per
        (input channel, output channel) pair.
      * batchnorm gamma (scale) and beta (shift): 2 per channel. (Its
        running mean/variance are *buffers*: tracked statistics, not trained.)
      * linear weights: n_features x n_classes, plus one bias per class.
    """
    lines = []
    for name, p in model.named_parameters():
        lines.append(f"  {name:28s} {str(tuple(p.shape)):22s} {p.numel():>9,}")
    total = count_parameters(model)
    clf = sum(p.numel() for p in model.classifier.parameters())
    lines.append(f"  {'TOTAL':28s} {'':22s} {total:>9,}")
    lines.append(
        f"  ({100 * clf / total:.0f}% are in the final linear layer: "
        f"{model.n_channels_out} channels x {model.n_freq_out} frequency rows "
        f"= {model.n_channels_out * model.n_freq_out} features -> {model.config.n_classes} classes)"
    )
    return "\n".join(lines)


@torch.no_grad()
def predict_logits(model: nn.Module, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Run the model on many inputs in eval mode (batchnorm uses its running
    statistics instead of the current batch's, so each prediction is
    independent of what else is in the batch)."""
    model.eval()
    return torch.cat([model(X[i : i + batch_size]) for i in range(0, len(X), batch_size)])


def load_checkpoint(path, device="cpu") -> tuple[TinyCNN, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TinyCNN(TinyCNNConfig.from_dict(ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


if __name__ == "__main__":
    m = TinyCNN(TinyCNNConfig())
    print(f"TinyCNN: {count_parameters(m):,} parameters")
    print(parameter_report(m))
