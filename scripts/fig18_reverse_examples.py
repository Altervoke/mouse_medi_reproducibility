"""Generate reverse-mechanism examples (Figure S4)."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper_figure_style import apply

ROOT = Path(__file__).resolve().parents[1]
MEDI_ROOT = ROOT / "data/assets"
OUT = ROOT / "figures/pdf/fig18_reverse_examples.pdf"
FRAME_IDS = list(range(8, 23, 2))
MECHANISMS = [("near-static candidate", "High retention, small stimulus change"),
              ("response-invariant despite large change", "High retention, large stimulus change")]


def find_gif(row):
    name = f"{int(row.session)}_{int(row.scan_idx)}_r{int(row.readout_id)}.gif"
    hits = list(MEDI_ROOT.glob(f"**/{name}"))
    return hits[0] if hits else None


def read_frames(path):
    image = Image.open(path)
    frames = []
    for index in FRAME_IDS:
        image.seek(min(index, getattr(image, "n_frames", 1) - 1))
        frames.append(np.asarray(image.convert("L")))
    return frames


def main():
    data = pd.read_csv(ROOT / "figures/tables/temporal_reverse_two_mechanisms.csv")
    chosen = {}
    for mechanism, _ in MECHANISMS:
        rows = []
        subset = data[data.mechanism.eq(mechanism)].sort_values(["median_retention", "mean_pixel_distance"], ascending=[False, False])
        for _, row in subset.iterrows():
            path = find_gif(row)
            if path is not None:
                rows.append((row, path))
            if len(rows) == 4:
                break
        if len(rows) != 4:
            raise RuntimeError(f"Need four available MEDIs for {mechanism}")
        chosen[mechanism] = rows
    apply()
    fig, axes_all = plt.subplots(9, 1, figsize=(10, 6.8), gridspec_kw={"left": .24, "right": .995, "top": .90, "bottom": .045, "hspace": .015, "height_ratios": [1, 1, 1, 1, .34, 1, 1, 1, 1]})
    axes = [axes_all[i] for i in (0, 1, 2, 3, 5, 6, 7, 8)]
    axes_all[4].axis("off")
    row_index = 0
    for mechanism, title in MECHANISMS:
        for local_index, (row, path) in enumerate(chosen[mechanism]):
            axis = axes[row_index]
            strip = np.concatenate(read_frames(path), axis=1)
            axis.imshow(strip, cmap="gray", vmin=0, vmax=255, aspect="equal", interpolation="nearest")
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.text(-.015, .5, f"{row.brain_area} | pixel distance {row.mean_pixel_distance:.3f}", transform=axis.transAxes, ha="right", va="center", fontsize=11)
            if row_index == 0:
                width = 256
                axis.set_xticks([(k + .5) * width for k in range(len(FRAME_IDS))], [str(k) for k in FRAME_IDS])
                axis.xaxis.tick_top(); axis.tick_params(axis="x", length=0, pad=1, labelsize=10)
                axis.text(-.015, 1.02, "frame", transform=axis.transAxes, ha="right", va="bottom", fontsize=10)
            else:
                axis.set_xticks([])
            row_index += 1
    for axis_index, (_, title) in zip((0, 4), MECHANISMS):
        box = axes[axis_index].get_position()
        offset = .060 if axis_index == 0 else .018
        fig.text(box.x0, box.y1 + offset, title, ha="left", va="bottom", fontsize=12)
    fig.savefig(OUT); plt.close(fig); print(OUT)


if __name__ == "__main__":
    main()
