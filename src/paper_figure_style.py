"""One visual specification for every regenerated manuscript figure."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

AREAS = ("V1", "LM", "RL", "AL")
AREA_COLORS = {"V1": "#2F5D8A", "LM": "#C9792B", "RL": "#A34F52", "AL": "#4A8F67"}
HEATMAP_CMAP = "vlag"


def apply() -> None:
    # Support both Linux/WSL and native Windows regeneration.  The
    # font lookup remains portable across those environments,
    # which can raise WinError 5 before any figure is written.
    _fonts = ("/mnt/c/Windows/Fonts/times.ttf", "/mnt/c/Windows/Fonts/timesbd.ttf",
              "/mnt/c/Windows/Fonts/timesi.ttf", "/mnt/c/Windows/Fonts/timesbi.ttf",
              r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\timesbd.ttf",
              r"C:\Windows\Fonts\timesi.ttf", r"C:\Windows\Fonts\timesbi.ttf")
    _times_path = None
    for _font in _fonts:
        try:
            if Path(_font).exists():
                font_manager.fontManager.addfont(_font)
                if _times_path is None and Path(_font).name == "times.ttf":
                    _times_path = _font
        except OSError:
            continue
    _times_name = "Times New Roman"
    if _times_path is not None:
        _times_name = font_manager.FontProperties(fname=_times_path).get_name()
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": _times_name, "font.serif": [_times_name],
        "mathtext.fontset": "stix", "pdf.fonttype": 42,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "legend.title_fontsize": 10,
        "axes.linewidth": .8, "lines.linewidth": 1.4, "lines.markersize": 5,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": .03,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def panel(ax, letter: str) -> None:
    ax.text(-.16, 1.05, letter, transform=ax.transAxes, fontweight="bold", va="bottom", fontsize=12)


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
