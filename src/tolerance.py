"""Canonical grayscale MEDI transformations and empirical-ellipse fitting.

The evaluator is independent of the licensed foundation-model adapter: callers
provide a function that maps a ``(time, height, width)`` clip to a response.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EmpiricalEllipse:
    center_xy: tuple[float, float]
    axes_xy: tuple[float, float]
    angle_deg: float
    threshold: float


def fit_empirical_ellipse(range_map: np.ndarray, threshold_fraction: float = 0.10) -> EmpiricalEllipse:
    """Fit a possibly tilted weighted ellipse to a single connected range map."""
    values = np.asarray(range_map, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("range_map must be a finite 2-D grayscale array")
    peak = float(values.max())
    threshold = threshold_fraction * peak
    weights = np.where(values >= threshold, np.maximum(values, 0.0), 0.0)
    yy, xx = np.indices(values.shape, dtype=np.float64)
    mass = float(weights.sum())
    if mass <= 0:
        center = ((values.shape[1] - 1) / 2.0, (values.shape[0] - 1) / 2.0)
        return EmpiricalEllipse(center, (1.0, 1.0), 0.0, threshold)
    cx = float((weights * xx).sum() / mass)
    cy = float((weights * yy).sum() / mass)
    dx, dy = xx - cx, yy - cy
    covariance = np.array([
        [(weights * dx * dx).sum() / mass, (weights * dx * dy).sum() / mass],
        [(weights * dx * dy).sum() / mass, (weights * dy * dy).sum() / mass],
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-12)
    principal = eigenvectors[:, order[0]]
    angle = float(np.degrees(np.arctan2(principal[1], principal[0])))
    return EmpiricalEllipse((cx, cy), tuple(np.sqrt(eigenvalues)), angle, threshold)


def empirical_ellipse_from_medi(medi: np.ndarray, threshold_fraction: float = 0.10) -> EmpiricalEllipse:
    clip = np.asarray(medi, dtype=np.float64)
    if clip.ndim != 3:
        raise ValueError("MEDI must have shape (time, height, width)")
    return fit_empirical_ellipse(clip.max(axis=0) - clip.min(axis=0), threshold_fraction)


def rotate_about_empirical_center(medi: np.ndarray, angle_deg: float) -> np.ndarray:
    clip = np.asarray(medi)
    cx, cy = empirical_ellipse_from_medi(clip).center_xy
    value_max = 1.0 if float(np.nanmax(clip)) <= 1.5 else 255.0
    frames = []
    for frame in clip:
        encoded = np.asarray(np.clip(frame / value_max * 255.0, 0, 255), dtype=np.uint8)
        image = Image.fromarray(encoded)
        rotated = image.rotate(float(angle_deg), resample=Image.Resampling.BICUBIC,
                               center=(cx, cy), fillcolor=int(np.asarray(encoded).mean()))
        frames.append(np.asarray(rotated, dtype=np.float32) / 255.0 * value_max)
    return np.stack(frames)


def translate_with_mean_padding(medi: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    clip = np.asarray(medi)
    result = np.empty_like(clip)
    height, width = clip.shape[-2:]
    for index, frame in enumerate(clip):
        shifted = np.full((height, width), frame.mean(), dtype=clip.dtype)
        xs0, xs1 = max(0, -dx), min(width, width - dx)
        ys0, ys1 = max(0, -dy), min(height, height - dy)
        xd0, xd1 = max(0, dx), min(width, width + dx)
        yd0, yd1 = max(0, dy), min(height, height + dy)
        if xs1 > xs0 and ys1 > ys0:
            shifted[yd0:yd1, xd0:xd1] = frame[ys0:ys1, xs0:xs1]
        result[index] = shifted
    return result


def temporal_resample(medi: np.ndarray, speed: float) -> np.ndarray:
    clip = np.asarray(medi, dtype=np.float32)
    if speed <= 0:
        raise ValueError("speed must be positive")
    count = int(np.floor((len(clip) - 1) / speed) + 1)
    positions = np.linspace(0, len(clip) - 1, max(2, count))
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(clip) - 1)
    weight = (positions - lower).reshape(-1, 1, 1)
    return clip[lower] * (1 - weight) + clip[upper] * weight


def transformed_clips(medi: np.ndarray) -> dict[str, np.ndarray]:
    clip = np.asarray(medi)
    mean = clip.mean(axis=(1, 2), keepdims=True)
    value_max = 1.0 if float(np.nanmax(clip)) <= 1.5 else 255.0
    variants: dict[str, np.ndarray] = {}
    for factor in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0):
        variants[f"contrast:{factor:g}"] = np.clip((clip - mean) * factor + mean, 0, value_max)
    for angle in (-2, -5, -10, -15, -30, -60, -90, 2, 5, 10, 15, 30, 60, 90):
        variants[f"rotation:{angle:g}"] = rotate_about_empirical_center(clip, angle)
    for distance in (1, 2, 4, 8, 16, 32, 48):
        for dx, dy, label in ((distance, 0, "x+"), (-distance, 0, "x-"),
                              (0, distance, "y+"), (0, -distance, "y-")):
            variants[f"translation:{label}{distance:g}"] = translate_with_mean_padding(clip, dx, dy)
    variants["temporal_reverse:-1"] = clip[::-1]
    for speed in (1.25, 1.5, 2.0, 0.5, 0.67, 0.8):
        variants[f"temporal_speed:{speed:g}"] = temporal_resample(clip, speed)
    return variants


def evaluate_retention(medi: np.ndarray, evaluate_clip: Callable[[np.ndarray], float]) -> list[dict[str, float | str]]:
    """Evaluate the original and all canonical transformations."""
    original = float(evaluate_clip(np.asarray(medi)))
    denominator = max(original, 1e-12)
    rows = [{"condition": "original", "parameter": 1.0, "response": original,
             "original_response": original, "retention": 1.0}]
    for key, clip in transformed_clips(medi).items():
        condition, parameter = key.split(":", 1)
        if condition == "translation":
            numeric = float(parameter[1:])
        elif condition == "temporal_reverse":
            numeric = -1.0
        else:
            numeric = float(parameter)
        if condition == "temporal_speed":
            condition = "speed_up" if numeric > 1 else "slow_down"
        response = float(evaluate_clip(clip))
        rows.append({"condition": condition, "parameter": numeric, "response": response,
                     "original_response": original, "retention": response / denominator})
    return rows
