"""Generate the transformation contact sheet (Figure S1)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper_figure_style import apply
from tolerance import rotate_about_empirical_center

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/assets/figS01_example.gif"
OUTPUT = ROOT / "figures/pdf/fig10_transform_examples.pdf"
FRAME_IDS = list(range(8, 23, 2))


def load_gif(path: Path) -> np.ndarray:
    image = Image.open(path)
    frames = []
    for index in range(getattr(image, "n_frames", 1)):
        image.seek(index)
        frames.append(np.asarray(image.convert("L"), dtype=np.float32))
    return np.stack(frames)


def transformed_examples(base: np.ndarray) -> dict[str, np.ndarray]:
    def shift(frame: np.ndarray, dx: int = 8) -> np.ndarray:
        result = np.full_like(frame, frame.mean())
        result[:, dx:] = frame[:, :-dx]
        return result

    def resample(sequence: np.ndarray, factor: float) -> np.ndarray:
        count = max(2, int(np.floor((len(sequence) - 1) / factor) + 1))
        positions = np.linspace(0, len(sequence) - 1, count)
        lower = np.floor(positions).astype(int)
        upper = np.minimum(lower + 1, len(sequence) - 1)
        weight = positions - lower
        return sequence[lower] * (1 - weight)[:, None, None] + sequence[upper] * weight[:, None, None]

    mean = base.mean((1, 2), keepdims=True)
    return {
        "original": base,
        "contrast (2.0x)": (base - mean) * 2 + mean,
        "rotation (+30 deg)": rotate_about_empirical_center(base, 30),
        "translation (+16 px x)": np.stack([shift(frame, 16) for frame in base]),
        "temporal reverse (-1)": base[::-1],
        "speed up (2.0x)": resample(base, 2),
        "slow down (0.5x)": resample(base, 0.5),
    }


def main() -> None:
    base = load_gif(SOURCE)
    variants = transformed_examples(base)
    apply()
    figure, axes = plt.subplots(len(variants), 1, figsize=(8, 6.2), sharex=False,
                                gridspec_kw={"left": 0.17, "right": 0.995, "top": 0.91,
                                             "bottom": 0.035, "hspace": 0.015})
    frame_width = base.shape[2]
    for row_index, (label, sequence) in enumerate(variants.items()):
        axis = axes[row_index]
        strip = np.concatenate([sequence[min(frame, len(sequence) - 1)] for frame in FRAME_IDS], axis=1)
        axis.imshow(strip, cmap="gray", vmin=0, vmax=255, aspect="equal", interpolation="nearest")
        axis.set_xlim(0, strip.shape[1]); axis.set_ylim(strip.shape[0], 0)
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.text(-0.015, 0.5, label, transform=axis.transAxes, ha="right", va="center", fontsize=11)
        if row_index == 0:
            axis.set_xticks([(i + 0.5) * frame_width for i in range(len(FRAME_IDS))],
                            [str(frame) for frame in FRAME_IDS])
            axis.xaxis.tick_top(); axis.tick_params(axis="x", length=0, pad=1, labelsize=10)
            axis.text(-0.015, 1.015, "frame", transform=axis.transAxes, ha="right", va="bottom", fontsize=10)
        else:
            axis.set_xticks([])
    figure.savefig(OUTPUT)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
