"""Single-file grayscale MEDI transformation API.

The only public entry point is :func:`transform_gif`.  It accepts one input
GIF, one canonical transformation name, and one canonical parameter, then
writes exactly one transformed GIF.  The accepted parameter sets are fixed to
the 55 variants used by the paper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]

_CONTRAST = (0.25, 0.5, 0.75, 1.25, 1.5, 2.0)
_ROTATION = (-2, -5, -10, -15, -30, -60, -90, 2, 5, 10, 15, 30, 60, 90)
_DISTANCES = (1, 2, 4, 8, 16, 32, 48)
_SPEED_UP = (1.25, 1.5, 2.0)
_SLOW_DOWN = (0.5, 0.67, 0.8)
_TRANSLATIONS = tuple(
    f"{axis}{sign}{distance}"
    for distance in _DISTANCES
    for axis, sign in (("x", "+"), ("x", "-"), ("y", "+"), ("y", "-"))
)


def _read_gif(path: PathLike) -> tuple[np.ndarray, int]:
    image = Image.open(path)
    duration = int(image.info.get("duration", 33))
    frames = []
    for index in range(getattr(image, "n_frames", 1)):
        image.seek(index)
        frames.append(np.asarray(image.convert("L"), dtype=np.float32) / 255.0)
    return np.stack(frames), duration


def _write_gif(path: PathLike, clip: np.ndarray, duration: int) -> None:
    frames = [Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8), mode="L")
              for frame in np.asarray(clip)]
    if not frames:
        raise ValueError("transformation produced no frames")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, duration=duration)


def _empirical_center(clip: np.ndarray) -> tuple[float, float]:
    range_map = clip.max(axis=0) - clip.min(axis=0)
    threshold = 0.10 * float(range_map.max())
    weights = np.where(range_map >= threshold, np.maximum(range_map, 0.0), 0.0)
    yy, xx = np.indices(range_map.shape, dtype=np.float64)
    mass = float(weights.sum())
    if mass <= 0:
        return (float((range_map.shape[1] - 1) / 2), float((range_map.shape[0] - 1) / 2))
    return (float((weights * xx).sum() / mass), float((weights * yy).sum() / mass))


def _rotate(clip: np.ndarray, angle: float) -> np.ndarray:
    cx, cy = _empirical_center(clip)
    return np.stack([
        np.asarray(Image.fromarray(np.clip(frame * 255, 0, 255).astype(np.uint8), mode="L")
                   .rotate(angle, resample=Image.Resampling.BICUBIC,
                           center=(cx, cy),
                           fillcolor=int(round(float(frame.mean()) * 255.0))),
                   dtype=np.float32) / 255.0
        for frame in clip
    ])


def _translate(clip: np.ndarray, axis: str, sign: str, distance: int) -> np.ndarray:
    shift_x = distance if axis == "x" and sign == "+" else -distance if axis == "x" else 0
    shift_y = distance if axis == "y" and sign == "+" else -distance if axis == "y" else 0
    height, width = clip.shape[1:]
    result = np.empty_like(clip)
    for index, frame in enumerate(clip):
        shifted = np.full((height, width), float(frame.mean()), dtype=np.float32)
        xs0, xs1 = max(0, -shift_x), min(width, width - shift_x)
        ys0, ys1 = max(0, -shift_y), min(height, height - shift_y)
        xd0, xd1 = max(0, shift_x), min(width, width + shift_x)
        yd0, yd1 = max(0, shift_y), min(height, height + shift_y)
        if xs1 > xs0 and ys1 > ys0:
            shifted[yd0:yd1, xd0:xd1] = frame[ys0:ys1, xs0:xs1]
        result[index] = shifted
    return result


def _resample(clip: np.ndarray, speed: float) -> np.ndarray:
    count = int(np.floor((len(clip) - 1) / speed) + 1)
    positions = np.linspace(0, len(clip) - 1, max(2, count))
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(clip) - 1)
    weight = (positions - lower).reshape(-1, 1, 1)
    return clip[lower] * (1 - weight) + clip[upper] * weight


def transform_gif(input_gif: PathLike, transformation: str, parameter, output_gif: PathLike) -> Path:
    """Transform one grayscale MEDI GIF using exactly one canonical variant.

    Examples::

        transform_gif("medi.gif", "contrast", 0.5, "out.gif")
        transform_gif("medi.gif", "rotation", -15, "out.gif")
        transform_gif("medi.gif", "translation", "x-16", "out.gif")
        transform_gif("medi.gif", "temporal_reverse", -1, "out.gif")
        transform_gif("medi.gif", "speed_up", 1.5, "out.gif")
        transform_gif("medi.gif", "slow_down", 0.67, "out.gif")

    Translation parameters must be one of ``x+1``, ``x-1``, ``y+1``,
    ``y-1`` (and the corresponding seven distances).  Any value outside the
    55 canonical variants raises ``ValueError``.
    """
    clip, duration = _read_gif(input_gif)
    name = str(transformation)
    if name == "contrast":
        value = float(parameter)
        if value not in _CONTRAST:
            raise ValueError(f"unsupported contrast parameter: {parameter}")
        mean = clip.mean(axis=(1, 2), keepdims=True)
        result = np.clip((clip - mean) * value + mean, 0, 1)
    elif name == "rotation":
        value = int(parameter)
        if value not in _ROTATION:
            raise ValueError(f"unsupported rotation parameter: {parameter}")
        result = _rotate(clip, value)
    elif name == "translation":
        token = str(parameter)
        if token not in _TRANSLATIONS:
            raise ValueError(f"unsupported translation parameter: {parameter}")
        result = _translate(clip, token[0], token[1], int(token[2:]))
    elif name == "temporal_reverse":
        if float(parameter) != -1.0:
            raise ValueError("temporal_reverse only accepts parameter -1")
        result = clip[::-1]
    elif name in ("speed_up", "slow_down"):
        value = float(parameter)
        allowed = _SPEED_UP if name == "speed_up" else _SLOW_DOWN
        if value not in allowed:
            raise ValueError(f"unsupported {name} parameter: {parameter}")
        result = _resample(clip, value)
    else:
        raise ValueError("transformation must be contrast, rotation, translation, temporal_reverse, speed_up, or slow_down")
    _write_gif(output_gif, result, duration)
    return Path(output_gif)


__all__ = ["transform_gif"]
