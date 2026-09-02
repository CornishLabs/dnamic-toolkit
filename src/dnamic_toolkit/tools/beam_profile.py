#!/usr/bin/env python3
"""Fit astigmatic laser-beam TIFF images with 2-D Gaussians.

The Gaussian convention used everywhere is

    I = background + amplitude * exp[-2 * ((u/w_u)^2 + (v/w_v)^2)]

so w_u and w_v are 1/e^2 INTENSITY radii. For this convention w = 2*sigma.

For every image the script performs:
  1. An x/y-aligned fit (theta = 0).
  2. A free-angle elliptical fit.
  3. A final fit with one common orthogonal principal-axis set, estimated from
     all free-angle fits.

After the image fits, each measured radius is fitted versus distance to the
Gaussian-beam propagation hyperbola

    w(z) = w0 * sqrt(1 + ((z - z0) / zR)^2)

where w0 is the minimum 1/e^2 intensity radius, z0 is the waist position, and
zR is the effective Rayleigh range. For an astigmatic beam, the two axes are
fitted independently and may therefore have different w0, z0, and zR values.

The final common-axis fitted centre is also plotted versus distance, and a
wide CSV records every Gaussian model parameter and its 1-sigma uncertainty.

The common axis-set angle is averaged with exp(4j*theta), not exp(2j*theta).
That makes the average invariant under theta -> theta + 90 degrees, which is
important for an astigmatic beam because the major and minor axes can swap as
z passes through the two foci.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.optimize import least_squares


SCRIPT_VERSION = "2026.09.01-propagation-uncertainty-v4"


# Parameter order used by the fit:
# fixed-angle: [background, amplitude, x0, y0, w1, w2]
# free-angle:  [background, amplitude, x0, y0, w1, w2, theta]


@dataclass
class FitResult:
    success: bool
    message: str
    params: np.ndarray
    errors: np.ndarray
    covariance: np.ndarray
    theta_fixed: Optional[float]
    r2: float
    rmse: float
    noise: float
    crop_bounds: tuple[int, int, int, int]
    n_points: int

    @property
    def theta(self) -> float:
        if self.theta_fixed is not None:
            return float(self.theta_fixed)
        return float(self.params[6])

    @property
    def background(self) -> float:
        return float(self.params[0])

    @property
    def amplitude(self) -> float:
        return float(self.params[1])

    @property
    def background_error(self) -> float:
        return float(self.errors[0])

    @property
    def amplitude_error(self) -> float:
        return float(self.errors[1])

    @property
    def x0(self) -> float:
        return float(self.params[2])

    @property
    def y0(self) -> float:
        return float(self.params[3])

    @property
    def x0_error(self) -> float:
        return float(self.errors[2])

    @property
    def y0_error(self) -> float:
        return float(self.errors[3])

    @property
    def w1(self) -> float:
        return float(self.params[4])

    @property
    def w2(self) -> float:
        return float(self.params[5])

    @property
    def w1_error(self) -> float:
        return float(self.errors[4])

    @property
    def w2_error(self) -> float:
        return float(self.errors[5])

    @property
    def theta_error(self) -> float:
        if self.theta_fixed is not None or len(self.errors) < 7:
            return float("nan")
        return float(self.errors[6])

    @property
    def ellipticity_strength(self) -> float:
        """0 for circular, approaching 1 for a very elongated fit."""
        denom = self.w1 + self.w2
        if denom <= 0:
            return 0.0
        return abs(self.w1 - self.w2) / denom

    @property
    def snr(self) -> float:
        return self.amplitude / max(self.noise, np.finfo(float).eps)


@dataclass
class ImageRecord:
    path: Path
    distance: float
    aligned: FitResult
    free: FitResult
    common: Optional[FitResult] = None
    angle_weight: float = 0.0


@dataclass
class PropagationFit:
    """Fit of a 1/e^2 radius to the Gaussian-beam propagation hyperbola."""

    success: bool
    message: str
    w0: float
    z0: float
    z_rayleigh: float
    w0_error: float
    z0_error: float
    z_rayleigh_error: float
    covariance: np.ndarray
    r2: float
    rmse: float
    reduced_chi2: float
    n_points: int

    @property
    def far_field_slope(self) -> float:
        """Asymptotic radius slope, w0/zR, in waist-unit per distance-unit."""
        if self.z_rayleigh <= 0:
            return float("nan")
        return self.w0 / self.z_rayleigh


@dataclass(frozen=True)
class BeamProfileSettings:
    """Inputs for a folder-level beam-profile analysis."""

    folder: Path | str
    output: Path | str | None = None
    centre_x: float = 671.0
    centre_y: float = 786.0
    fit_half_size: int = 180
    fit_stride: int = 1
    pixel_size: float = 1.0
    waist_unit: str = "pixel"
    distance_unit: str = "filename unit"
    min_angle_r2: float = 0.80
    robust: bool = False


@dataclass(frozen=True)
class BeamProfileResult:
    """Output paths and fit summaries from a beam-profile analysis."""

    records: tuple[ImageRecord, ...]
    output: Path
    diagnostics: Path
    results_csv: Path
    propagation_csv: Path
    centre_plot: Path
    waists_xy_plot: Path
    waists_principal_axes_plot: Path
    common_theta: float
    common_angle_coherence: float
    common_angle_effective_n: float
    aligned_propagation: tuple[PropagationFit, PropagationFit]
    common_propagation: tuple[PropagationFit, PropagationFit]

    @property
    def successful_common_fits(self) -> int:
        return sum(
            record.common is not None and record.common.success
            for record in self.records
        )


def canonical_axis_set_angle(theta: float) -> float:
    """Map an orthogonal axis-set angle into [-pi/4, pi/4)."""
    return float((theta + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)


def canonical_free_angle(theta: float) -> float:
    """Map a single directed-axis angle into [-pi/2, pi/2)."""
    return float((theta + np.pi / 2.0) % np.pi - np.pi / 2.0)


def gaussian_2d(
    x: np.ndarray,
    y: np.ndarray,
    params: Sequence[float],
    theta_fixed: Optional[float] = None,
) -> np.ndarray:
    """Rotated 2-D Gaussian with 1/e^2 radii w1 and w2."""
    background, amplitude, x0, y0, w1, w2 = params[:6]
    theta = float(theta_fixed) if theta_fixed is not None else float(params[6])

    c = np.cos(theta)
    s = np.sin(theta)
    dx = x - x0
    dy = y - y0

    # u is at theta from +x; v is its orthogonal partner.
    u = c * dx + s * dy
    v = -s * dx + c * dy

    exponent = -2.0 * ((u / w1) ** 2 + (v / w2) ** 2)
    return background + amplitude * np.exp(exponent)


def read_tiff_2d(path: Path) -> np.ndarray:
    image = np.asarray(tifffile.imread(path))
    image = np.squeeze(image)

    if image.ndim == 3 and image.shape[-1] in (3, 4):
        # Gracefully handle RGB/RGBA images, though beam cameras are usually mono.
        image = image[..., :3].mean(axis=-1)
    elif image.ndim > 2:
        # Treat the first plane as the image for an unexpected TIFF stack.
        image = np.asarray(image[0])
        image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D TIFF, got shape {image.shape} in {path}")

    return image.astype(np.float64, copy=False)


def numeric_stem(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError as exc:
        raise ValueError(
            f"TIFF filename stem must be numeric, e.g. 205.5.tif; got {path.name}"
        ) from exc


def find_tiffs(folder: Path) -> list[Path]:
    paths = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ]
    valid: list[tuple[float, Path]] = []
    for path in paths:
        try:
            valid.append((numeric_stem(path), path))
        except ValueError as exc:
            print(f"Warning: {exc}; skipping.", file=sys.stderr)
    valid.sort(key=lambda item: item[0])
    return [p for _, p in valid]


def crop_bounds(
    shape: tuple[int, int], center_x: float, center_y: float, half_size: int
) -> tuple[int, int, int, int]:
    height, width = shape
    x_lo = max(0, int(math.floor(center_x - half_size)))
    x_hi = min(width, int(math.ceil(center_x + half_size + 1)))
    y_lo = max(0, int(math.floor(center_y - half_size)))
    y_hi = min(height, int(math.ceil(center_y + half_size + 1)))
    if x_hi - x_lo < 8 or y_hi - y_lo < 8:
        raise ValueError("Fit crop is too small or lies outside the image.")
    return x_lo, x_hi, y_lo, y_hi


def border_pixels(array: np.ndarray) -> np.ndarray:
    if min(array.shape) < 4:
        return array.ravel()
    return np.concatenate(
        [array[0, :], array[-1, :], array[1:-1, 0], array[1:-1, -1]]
    )


def robust_noise_and_background(array: np.ndarray) -> tuple[float, float]:
    border = border_pixels(array)
    background = float(np.median(border))
    mad = float(np.median(np.abs(border - background)))
    noise = 1.4826 * mad
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(border))
    if not np.isfinite(noise) or noise <= 0:
        noise = max(float(np.ptp(array)) * 1e-6, 1.0)
    return background, noise


def moment_initial_guess(
    crop: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    theta_fixed: Optional[float],
) -> tuple[np.ndarray, float]:
    background, noise = robust_noise_and_background(crop)
    peak = float(np.max(crop))
    amplitude = max(peak - background, noise)

    signal = crop - background
    # Suppress low-level background fluctuations in the moment estimate only.
    threshold = max(3.0 * noise, 0.02 * amplitude)
    weights = np.clip(signal - threshold, 0.0, None)
    if np.sum(weights) <= 0:
        weights = np.clip(signal, 0.0, None)
    if np.sum(weights) <= 0:
        weights = np.ones_like(crop)

    xx, yy = np.meshgrid(x_coords, y_coords)
    total = float(np.sum(weights))
    x0 = float(np.sum(weights * xx) / total)
    y0 = float(np.sum(weights * yy) / total)

    dx = xx - x0
    dy = yy - y0
    cov_xx = float(np.sum(weights * dx * dx) / total)
    cov_yy = float(np.sum(weights * dy * dy) / total)
    cov_xy = float(np.sum(weights * dx * dy) / total)
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)

    # Ensure a positive semidefinite covariance estimate.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.25, None)
    covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    if theta_fixed is None:
        major_index = int(np.argmax(eigenvalues))
        vector = eigenvectors[:, major_index]
        theta = canonical_free_angle(math.atan2(vector[1], vector[0]))
    else:
        theta = float(theta_fixed)

    c = math.cos(theta)
    s = math.sin(theta)
    u = np.array([c, s])
    v = np.array([-s, c])
    variance_u = max(float(u @ covariance @ u), 0.25)
    variance_v = max(float(v @ covariance @ v), 0.25)

    # For exp(-2*x^2/w^2), variance sigma^2 = w^2/4, hence w = 2*sigma.
    w1 = 2.0 * math.sqrt(variance_u)
    w2 = 2.0 * math.sqrt(variance_v)

    if theta_fixed is None:
        p0 = np.array([background, amplitude, x0, y0, w1, w2, theta])
    else:
        p0 = np.array([background, amplitude, x0, y0, w1, w2])
    return p0, noise


def fit_image(
    image: np.ndarray,
    center_guess: tuple[float, float],
    half_size: int,
    theta_fixed: Optional[float],
    fit_stride: int,
    robust_loss: bool,
) -> FitResult:
    x_lo, x_hi, y_lo, y_hi = crop_bounds(
        image.shape, center_guess[0], center_guess[1], half_size
    )
    crop = image[y_lo:y_hi, x_lo:x_hi]
    x_full = np.arange(x_lo, x_hi, dtype=float)
    y_full = np.arange(y_lo, y_hi, dtype=float)

    p0, noise = moment_initial_guess(crop, x_full, y_full, theta_fixed)

    stride = max(1, int(fit_stride))
    fit_crop = crop[::stride, ::stride]
    x_coords = x_full[::stride]
    y_coords = y_full[::stride]
    xx, yy = np.meshgrid(x_coords, y_coords)
    z = fit_crop.ravel()

    data_min = float(np.min(fit_crop))
    data_max = float(np.max(fit_crop))
    data_span = max(data_max - data_min, noise)
    max_width = 4.0 * max(crop.shape)

    if theta_fixed is None:
        lower = np.array(
            [
                data_min - 2.0 * data_span,
                0.0,
                x_lo,
                y_lo,
                0.5,
                0.5,
                -np.pi / 2.0,
            ]
        )
        upper = np.array(
            [
                data_max + 2.0 * data_span,
                np.inf,
                x_hi - 1.0,
                y_hi - 1.0,
                max_width,
                max_width,
                np.pi / 2.0,
            ]
        )
    else:
        lower = np.array(
            [data_min - 2.0 * data_span, 0.0, x_lo, y_lo, 0.5, 0.5]
        )
        upper = np.array(
            [
                data_max + 2.0 * data_span,
                np.inf,
                x_hi - 1.0,
                y_hi - 1.0,
                max_width,
                max_width,
            ]
        )

    # Keep the initial point strictly inside the bounds.
    finite_upper = np.where(np.isfinite(upper), upper, p0 + 1e12)
    margin = 1e-9
    p0 = np.maximum(p0, lower + margin)
    p0 = np.minimum(p0, finite_upper - margin)

    def residuals(params: np.ndarray) -> np.ndarray:
        model = gaussian_2d(xx, yy, params, theta_fixed=theta_fixed)
        return (model.ravel() - z) / noise

    try:
        result = least_squares(
            residuals,
            p0,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            loss="soft_l1" if robust_loss else "linear",
            f_scale=1.0,
            max_nfev=4000,
        )

        params = result.x.astype(float)
        if theta_fixed is None:
            params[6] = canonical_free_angle(params[6])

        model = gaussian_2d(xx, yy, params, theta_fixed=theta_fixed).ravel()
        residual_raw = z - model
        ss_res = float(np.sum(residual_raw**2))
        ss_tot = float(np.sum((z - np.mean(z)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = math.sqrt(ss_res / max(z.size, 1))

        n_parameters = len(params)
        dof = max(z.size - n_parameters, 1)
        covariance = np.full((n_parameters, n_parameters), np.nan)
        errors = np.full(n_parameters, np.nan)
        if result.jac.size > 0:
            try:
                # The residuals are normalized by the border-noise estimate.
                # Treat that as an absolute lower bound on the per-pixel noise,
                # while allowing excess residual scatter/model mismatch to
                # inflate the parameter covariance.
                jtj_inv = np.linalg.pinv(result.jac.T @ result.jac)
                reduced_chi2 = float(np.sum(result.fun**2) / dof)
                covariance = jtj_inv * max(reduced_chi2, 1.0)
                diagonal = np.diag(covariance)
                errors = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))
            except np.linalg.LinAlgError:
                pass

        success = bool(result.success and np.all(np.isfinite(params)))
        return FitResult(
            success=success,
            message=result.message,
            params=params,
            errors=errors,
            covariance=covariance,
            theta_fixed=theta_fixed,
            r2=r2,
            rmse=rmse,
            noise=noise,
            crop_bounds=(x_lo, x_hi, y_lo, y_hi),
            n_points=z.size,
        )
    except Exception as exc:  # Keep processing the rest of the folder.
        n_parameters = 7 if theta_fixed is None else 6
        return FitResult(
            success=False,
            message=f"{type(exc).__name__}: {exc}",
            params=np.full(n_parameters, np.nan),
            errors=np.full(n_parameters, np.nan),
            covariance=np.full((n_parameters, n_parameters), np.nan),
            theta_fixed=theta_fixed,
            r2=float("nan"),
            rmse=float("nan"),
            noise=noise,
            crop_bounds=(x_lo, x_hi, y_lo, y_hi),
            n_points=z.size,
        )


def free_fit_angle_weight(fit: FitResult, min_r2: float) -> float:
    """Weight an individual free-angle fit for the common axis-set average.

    Weight = ellipticity^2 * R2_quality^2 * SNR_quality^2 / angle_error^2.

    Nearly circular beams carry little orientation information, so their
    ellipticity factor tends to zero. The angle uncertainty and goodness of fit
    provide the requested confidence weighting.
    """
    if not fit.success:
        return 0.0
    if not all(np.isfinite([fit.w1, fit.w2, fit.theta, fit.r2, fit.snr])):
        return 0.0

    ellipticity = fit.ellipticity_strength
    if ellipticity <= 0:
        return 0.0

    if min_r2 >= 1.0:
        r2_quality = 1.0 if fit.r2 >= min_r2 else 0.0
    else:
        r2_quality = float(np.clip((fit.r2 - min_r2) / (1.0 - min_r2), 0, 1))

    # Saturate this factor once the peak is comfortably above the noise.
    snr_quality = float(np.clip(fit.snr / 20.0, 0, 1))

    theta_error = fit.theta_error
    if not np.isfinite(theta_error) or theta_error <= 0:
        theta_error = np.deg2rad(30.0)
    theta_error = max(theta_error, np.deg2rad(0.25))

    return (
        ellipticity**2
        * r2_quality**2
        * snr_quality**2
        / theta_error**2
    )


def average_axis_set(
    records: Sequence[ImageRecord], min_r2: float
) -> tuple[float, float, float]:
    """Return common angle, 4-theta resultant coherence, and effective N."""
    phasors: list[complex] = []
    weights: list[float] = []

    for record in records:
        weight = free_fit_angle_weight(record.free, min_r2=min_r2)
        record.angle_weight = weight
        if weight > 0 and np.isfinite(record.free.theta):
            phasors.append(np.exp(4j * record.free.theta))
            weights.append(weight)

    if not weights:
        # Last-resort fallback: equal-weight average of any successful free fits.
        for record in records:
            if record.free.success and np.isfinite(record.free.theta):
                phasors.append(np.exp(4j * record.free.theta))
                weights.append(1.0)

    if not weights:
        raise RuntimeError("No successful free-angle fits were available.")

    weights_array = np.asarray(weights, dtype=float)
    phasor_array = np.asarray(phasors, dtype=complex)
    vector = np.sum(weights_array * phasor_array)
    theta = canonical_axis_set_angle(0.25 * np.angle(vector))
    coherence = float(abs(vector) / np.sum(weights_array))
    effective_n = float(np.sum(weights_array) ** 2 / np.sum(weights_array**2))
    return theta, coherence, effective_n


def projected_sigmas(fit: FitResult) -> tuple[float, float]:
    """Gaussian sigma projected on image x and y; w = 2*sigma."""
    sigma_u = fit.w1 / 2.0
    sigma_v = fit.w2 / 2.0
    c = math.cos(fit.theta)
    s = math.sin(fit.theta)
    sigma_x = math.sqrt((c * sigma_u) ** 2 + (s * sigma_v) ** 2)
    sigma_y = math.sqrt((s * sigma_u) ** 2 + (c * sigma_v) ** 2)
    return sigma_x, sigma_y


def diagnostic_plot(
    image: np.ndarray,
    record: ImageRecord,
    output_path: Path,
    pixel_size: float,
    waist_unit: str,
    distance_unit: str,
    common_theta: float,
) -> None:
    fit = record.common
    if fit is None or not fit.success:
        return

    sigma_x, sigma_y = projected_sigmas(fit)
    height, width = image.shape
    x_lo = max(0, int(math.floor(fit.x0 - 5.0 * sigma_x)))
    x_hi = min(width, int(math.ceil(fit.x0 + 5.0 * sigma_x + 1)))
    y_lo = max(0, int(math.floor(fit.y0 - 5.0 * sigma_y)))
    y_hi = min(height, int(math.ceil(fit.y0 + 5.0 * sigma_y + 1)))

    # Guarantee a useful view even for unusually narrow fits.
    if x_hi - x_lo < 12:
        x_lo = max(0, int(round(fit.x0)) - 6)
        x_hi = min(width, x_lo + 12)
    if y_hi - y_lo < 12:
        y_lo = max(0, int(round(fit.y0)) - 6)
        y_hi = min(height, y_lo + 12)

    crop = image[y_lo:y_hi, x_lo:x_hi]
    x_coords = np.arange(x_lo, x_hi, dtype=float)
    y_coords = np.arange(y_lo, y_hi, dtype=float)

    row_index = int(np.clip(round(fit.y0), 0, height - 1))
    col_index = int(np.clip(round(fit.x0), 0, width - 1))
    x_raw = image[row_index, x_lo:x_hi]
    y_raw = image[y_lo:y_hi, col_index]

    x_fit = gaussian_2d(
        x_coords,
        np.full_like(x_coords, float(row_index)),
        fit.params,
        theta_fixed=common_theta,
    )
    y_fit = gaussian_2d(
        np.full_like(y_coords, float(col_index)),
        y_coords,
        fit.params,
        theta_fixed=common_theta,
    )

    fig = plt.figure(figsize=(9.5, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(5.0, 1.35),
        height_ratios=(1.35, 5.0),
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_image = fig.add_subplot(grid[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax_image)
    ax_blank = fig.add_subplot(grid[0, 1])
    ax_blank.axis("off")

    low, high = np.nanpercentile(crop, [0.5, 99.8])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(crop)), float(np.nanmax(crop))

    ax_image.imshow(
        crop,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        extent=(x_lo - 0.5, x_hi - 0.5, y_hi - 0.5, y_lo - 0.5),
        vmin=low,
        vmax=high,
    )
    ax_image.plot(fit.x0, fit.y0, marker="+", markersize=12, linestyle="none")

    # Draw the two common principal axes out to +/-5 sigma = +/-2.5w.
    for theta, waist in ((common_theta, fit.w1), (common_theta + np.pi / 2, fit.w2)):
        half_length = 2.5 * waist
        dx = half_length * math.cos(theta)
        dy = half_length * math.sin(theta)
        ax_image.plot(
            [fit.x0 - dx, fit.x0 + dx],
            [fit.y0 - dy, fit.y0 + dy],
            linestyle="--",
            linewidth=1.2,
        )

    ax_top.plot(x_coords, x_raw, ".", markersize=2.5, label="raw centre-row slice")
    ax_top.plot(x_coords, x_fit, linewidth=1.5, label="2-D Gaussian on slice")
    ax_top.set_ylabel("Intensity [camera units]")
    ax_top.legend(loc="best", fontsize=8)
    ax_top.tick_params(labelbottom=False)

    ax_right.plot(y_raw, y_coords, ".", markersize=2.5)
    ax_right.plot(y_fit, y_coords, linewidth=1.5)
    ax_right.set_xlabel("Intensity\n[camera units]")
    ax_right.tick_params(labelleft=False)

    ax_image.set_xlabel("Camera x [pixel]")
    ax_image.set_ylabel("Camera y [pixel]")
    ax_image.set_xlim(x_lo - 0.5, x_hi - 0.5)
    ax_image.set_ylim(y_hi - 0.5, y_lo - 0.5)

    w1 = fit.w1 * pixel_size
    w2 = fit.w2 * pixel_size
    e1 = fit.w1_error * pixel_size
    e2 = fit.w2_error * pixel_size
    angle_deg = np.rad2deg(common_theta)
    title = (
        f"{record.path.name}   distance={record.distance:g} {distance_unit}\n"
        f"common-axis fit: theta={angle_deg:.3f} deg, "
        f"w_u={w1:.4g}±{e1:.2g} {waist_unit}, "
        f"w_v={w2:.4g}±{e2:.2g} {waist_unit}; "
        f"1/e² intensity radii; R²={fit.r2:.6f}"
    )
    fig.suptitle(title, fontsize=11)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def safe_column_unit(unit: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", unit.strip()).strip("_")
    return cleaned or "units"


def _fit_parameter_columns(prefix: str, waist_unit: str) -> list[str]:
    """Column names for one complete 2-D Gaussian fit."""
    wu = safe_column_unit(waist_unit)
    return [
        f"{prefix}_fit_success",
        f"{prefix}_fit_message",
        f"{prefix}_background_camera_units",
        f"{prefix}_background_error_camera_units",
        f"{prefix}_amplitude_camera_units",
        f"{prefix}_amplitude_error_camera_units",
        f"{prefix}_x0_pixel",
        f"{prefix}_x0_error_pixel",
        f"{prefix}_y0_pixel",
        f"{prefix}_y0_error_pixel",
        f"{prefix}_w1_1e2_radius_{wu}",
        f"{prefix}_w1_error_{wu}",
        f"{prefix}_w2_1e2_radius_{wu}",
        f"{prefix}_w2_error_{wu}",
        f"{prefix}_theta_deg",
        f"{prefix}_theta_error_deg",
        f"{prefix}_r2",
        f"{prefix}_rmse_camera_units",
        f"{prefix}_noise_camera_units",
        f"{prefix}_snr",
        f"{prefix}_ellipticity_strength",
        f"{prefix}_n_fit_points",
        f"{prefix}_crop_x_low_pixel",
        f"{prefix}_crop_x_high_exclusive_pixel",
        f"{prefix}_crop_y_low_pixel",
        f"{prefix}_crop_y_high_exclusive_pixel",
    ]


def _fit_parameter_row(
    prefix: str,
    fit: Optional[FitResult],
    pixel_size: float,
) -> dict[str, object]:
    """Serialize every model parameter and its standard error for one fit."""
    if fit is None:
        return {
            f"{prefix}_fit_success": False,
            f"{prefix}_fit_message": "fit not available",
            f"{prefix}_background_camera_units": np.nan,
            f"{prefix}_background_error_camera_units": np.nan,
            f"{prefix}_amplitude_camera_units": np.nan,
            f"{prefix}_amplitude_error_camera_units": np.nan,
            f"{prefix}_x0_pixel": np.nan,
            f"{prefix}_x0_error_pixel": np.nan,
            f"{prefix}_y0_pixel": np.nan,
            f"{prefix}_y0_error_pixel": np.nan,
            f"{prefix}_w1_1e2_radius_UNIT": np.nan,
            f"{prefix}_w1_error_UNIT": np.nan,
            f"{prefix}_w2_1e2_radius_UNIT": np.nan,
            f"{prefix}_w2_error_UNIT": np.nan,
            f"{prefix}_theta_deg": np.nan,
            f"{prefix}_theta_error_deg": np.nan,
            f"{prefix}_r2": np.nan,
            f"{prefix}_rmse_camera_units": np.nan,
            f"{prefix}_noise_camera_units": np.nan,
            f"{prefix}_snr": np.nan,
            f"{prefix}_ellipticity_strength": np.nan,
            f"{prefix}_n_fit_points": 0,
            f"{prefix}_crop_x_low_pixel": np.nan,
            f"{prefix}_crop_x_high_exclusive_pixel": np.nan,
            f"{prefix}_crop_y_low_pixel": np.nan,
            f"{prefix}_crop_y_high_exclusive_pixel": np.nan,
        }

    x_lo, x_hi, y_lo, y_hi = fit.crop_bounds
    theta_error_deg = (
        np.rad2deg(fit.theta_error) if np.isfinite(fit.theta_error) else np.nan
    )
    return {
        f"{prefix}_fit_success": fit.success,
        f"{prefix}_fit_message": fit.message,
        f"{prefix}_background_camera_units": fit.background,
        f"{prefix}_background_error_camera_units": fit.background_error,
        f"{prefix}_amplitude_camera_units": fit.amplitude,
        f"{prefix}_amplitude_error_camera_units": fit.amplitude_error,
        f"{prefix}_x0_pixel": fit.x0,
        f"{prefix}_x0_error_pixel": fit.x0_error,
        f"{prefix}_y0_pixel": fit.y0,
        f"{prefix}_y0_error_pixel": fit.y0_error,
        # The caller renames the generic waist-unit suffix below.
        f"{prefix}_w1_1e2_radius_UNIT": fit.w1 * pixel_size,
        f"{prefix}_w1_error_UNIT": fit.w1_error * pixel_size,
        f"{prefix}_w2_1e2_radius_UNIT": fit.w2 * pixel_size,
        f"{prefix}_w2_error_UNIT": fit.w2_error * pixel_size,
        f"{prefix}_theta_deg": np.rad2deg(fit.theta),
        f"{prefix}_theta_error_deg": theta_error_deg,
        f"{prefix}_r2": fit.r2,
        f"{prefix}_rmse_camera_units": fit.rmse,
        f"{prefix}_noise_camera_units": fit.noise,
        f"{prefix}_snr": fit.snr,
        f"{prefix}_ellipticity_strength": fit.ellipticity_strength,
        f"{prefix}_n_fit_points": fit.n_points,
        f"{prefix}_crop_x_low_pixel": x_lo,
        f"{prefix}_crop_x_high_exclusive_pixel": x_hi,
        f"{prefix}_crop_y_low_pixel": y_lo,
        f"{prefix}_crop_y_high_exclusive_pixel": y_hi,
    }


def write_csv(
    records: Sequence[ImageRecord],
    output_path: Path,
    common_theta: float,
    pixel_size: float,
    waist_unit: str,
    distance_unit: str,
) -> None:
    """Write all per-image fit parameters and 1-sigma uncertainties.

    This is deliberately a wide, analysis-ready CSV: each row represents one
    TIFF and contains the complete aligned, free-angle, and common-axis fit.
    Camera centres remain in pixels so separate runs from the same camera can
    be compared directly without requiring a physical pixel calibration.
    """
    wu = safe_column_unit(waist_unit)
    du = safe_column_unit(distance_unit)
    columns = [
        "filename",
        f"distance_{du}",
        "pixel_size_applied",
        "waist_unit",
        "distance_unit",
        "common_axis_u_angle_deg",
        "common_axis_v_angle_deg",
        "free_angle_average_weight",
    ]
    for prefix in ("aligned", "free", "common"):
        columns.extend(_fit_parameter_columns(prefix, waist_unit))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row: dict[str, object] = {
                "filename": record.path.name,
                f"distance_{du}": record.distance,
                "pixel_size_applied": pixel_size,
                "waist_unit": waist_unit,
                "distance_unit": distance_unit,
                "common_axis_u_angle_deg": np.rad2deg(common_theta),
                "common_axis_v_angle_deg": np.rad2deg(common_theta + np.pi / 2.0),
                "free_angle_average_weight": record.angle_weight,
            }
            for prefix, fit in (
                ("aligned", record.aligned),
                ("free", record.free),
                ("common", record.common),
            ):
                values = _fit_parameter_row(prefix, fit, pixel_size)
                # Substitute the actual waist-unit suffix in the four radius keys.
                for key, value in values.items():
                    row[key.replace("_UNIT", f"_{wu}")] = value
            writer.writerow(row)


def finite_plot_arrays(
    records: Sequence[ImageRecord],
    fit_selector,
    pixel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances = []
    w1_values = []
    w1_errors = []
    w2_values = []
    w2_errors = []
    for record in records:
        fit = fit_selector(record)
        if fit is None or not fit.success:
            continue
        values = [record.distance, fit.w1, fit.w2]
        if not np.all(np.isfinite(values)):
            continue
        distances.append(record.distance)
        w1_values.append(fit.w1 * pixel_size)
        w2_values.append(fit.w2 * pixel_size)
        w1_errors.append(fit.w1_error * pixel_size)
        w2_errors.append(fit.w2_error * pixel_size)
    return tuple(
        np.asarray(values, dtype=float)
        for values in (distances, w1_values, w1_errors, w2_values, w2_errors)
    )



def finite_centre_arrays(
    records: Sequence[ImageRecord],
    fit_selector,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return distance, x0, x0 error, y0, and y0 error for successful fits."""
    distances: list[float] = []
    x_values: list[float] = []
    x_errors: list[float] = []
    y_values: list[float] = []
    y_errors: list[float] = []
    for record in records:
        fit = fit_selector(record)
        if fit is None or not fit.success:
            continue
        if not np.all(np.isfinite([record.distance, fit.x0, fit.y0])):
            continue
        distances.append(record.distance)
        x_values.append(fit.x0)
        y_values.append(fit.y0)
        x_errors.append(fit.x0_error)
        y_errors.append(fit.y0_error)
    return tuple(
        np.asarray(values, dtype=float)
        for values in (distances, x_values, x_errors, y_values, y_errors)
    )


def plot_centres(
    records: Sequence[ImageRecord],
    output_path: Path,
    distance_unit: str,
) -> None:
    """Plot the common-axis fitted camera centre versus propagation position."""
    distance, x0, x0_error, y0, y0_error = finite_centre_arrays(
        records, fit_selector=lambda record: record.common
    )
    if distance.size == 0:
        return

    order = np.argsort(distance)
    distance, x0, x0_error, y0, y0_error = (
        values[order] for values in (distance, x0, x0_error, y0, y0_error)
    )

    # Matplotlib rejects arrays containing only invalid error bars. Omitting the
    # error bars in that rare case is more useful than failing the whole script.
    xerr = x0_error if np.any(np.isfinite(x0_error) & (x0_error >= 0)) else None
    yerr = y0_error if np.any(np.isfinite(y0_error) & (y0_error >= 0)) else None

    fig, axes = plt.subplots(
        2, 1, figsize=(9.4, 7.0), sharex=True, constrained_layout=True
    )
    axes[0].errorbar(
        distance,
        x0,
        yerr=xerr,
        marker="o",
        linestyle="-",
        capsize=3,
        label="fitted x centre",
    )
    axes[1].errorbar(
        distance,
        y0,
        yerr=yerr,
        marker="o",
        linestyle="-",
        capsize=3,
        label="fitted y centre",
    )
    axes[0].set_ylabel("Camera centre x₀ [pixel]")
    axes[1].set_ylabel("Camera centre y₀ [pixel]")
    axes[1].set_xlabel(f"Distance from TIFF filename [{distance_unit}]")
    axes[0].set_title("Beam centre from final common-principal-axis Gaussian fits")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def gaussian_beam_radius(
    distance: np.ndarray | float,
    w0: float,
    z0: float,
    z_rayleigh: float,
) -> np.ndarray:
    """Gaussian-beam 1/e^2 radius as a function of propagation distance."""
    z = np.asarray(distance, dtype=float)
    return w0 * np.sqrt(1.0 + ((z - z0) / z_rayleigh) ** 2)


def gaussian_beam_radius_standard_error(
    distance: np.ndarray | float,
    w0: float,
    z0: float,
    z_rayleigh: float,
    covariance: np.ndarray,
) -> np.ndarray:
    """Propagate parameter covariance to the fitted 1/e² radius.

    This uses the first-order covariance propagation ``J C J.T``, where ``J``
    is the gradient of :func:`gaussian_beam_radius` with respect to
    ``(w0, z0, z_rayleigh)``.
    """

    z = np.asarray(distance, dtype=float)
    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.shape != (3, 3) or not np.all(np.isfinite(covariance_array)):
        return np.full_like(z, np.nan, dtype=float)
    if not np.all(np.isfinite([w0, z0, z_rayleigh])) or z_rayleigh <= 0:
        return np.full_like(z, np.nan, dtype=float)

    scaled_distance = (z - z0) / z_rayleigh
    root = np.sqrt(1.0 + scaled_distance**2)
    jacobian = np.stack(
        (
            root,
            -w0 * scaled_distance / (z_rayleigh * root),
            -w0 * scaled_distance**2 / (z_rayleigh * root),
        ),
        axis=-1,
    )
    variance = np.einsum("...i,ij,...j->...", jacobian, covariance_array, jacobian)
    return np.sqrt(np.maximum(variance, 0.0))


def failed_propagation_fit(message: str, n_points: int) -> PropagationFit:
    return PropagationFit(
        success=False,
        message=message,
        w0=float("nan"),
        z0=float("nan"),
        z_rayleigh=float("nan"),
        w0_error=float("nan"),
        z0_error=float("nan"),
        z_rayleigh_error=float("nan"),
        covariance=np.full((3, 3), np.nan),
        r2=float("nan"),
        rmse=float("nan"),
        reduced_chi2=float("nan"),
        n_points=n_points,
    )


def fit_gaussian_beam_propagation(
    distance: np.ndarray,
    waist: np.ndarray,
    waist_error: np.ndarray,
    robust_loss: bool,
) -> PropagationFit:
    """Fit w(z) = w0*sqrt(1 + ((z-z0)/zR)^2).

    Positive scale parameters are optimized in logarithmic form. Reported
    covariance is transformed back into the physical parameters (w0, z0, zR).
    """
    z = np.asarray(distance, dtype=float)
    w = np.asarray(waist, dtype=float)
    e = np.asarray(waist_error, dtype=float)

    finite = np.isfinite(z) & np.isfinite(w) & (w > 0)
    z = z[finite]
    w = w[finite]
    e = e[finite]

    if z.size < 3:
        return failed_propagation_fit(
            "At least three finite waist measurements are required.", z.size
        )
    if np.unique(z).size < 3:
        return failed_propagation_fit(
            "At least three distinct distance values are required.", z.size
        )

    order = np.argsort(z)
    z, w, e = z[order], w[order], e[order]
    z_span = float(np.ptp(z))
    if not np.isfinite(z_span) or z_span <= 0:
        return failed_propagation_fit("Distance span must be positive.", z.size)

    # Use measured fit uncertainties when available. Invalid uncertainties are
    # replaced with the median valid uncertainty. If none are usable, perform
    # an unweighted fit and scale the covariance by the residual variance.
    valid_errors = np.isfinite(e) & (e > 0)
    uses_measured_errors = bool(np.count_nonzero(valid_errors) >= 2)
    if uses_measured_errors:
        replacement = float(np.median(e[valid_errors]))
        sigma = np.where(valid_errors, e, replacement)
        # Prevent a single unrealistically tiny image-fit error dominating the
        # entire propagation fit.
        sigma_floor = max(replacement * 1e-3, np.finfo(float).eps)
        sigma = np.maximum(sigma, sigma_floor)
    else:
        sigma = np.ones_like(w)

    minimum_index = int(np.argmin(w))
    z0_guess = float(z[minimum_index])
    w0_guess = max(float(w[minimum_index]) * 0.98, np.finfo(float).tiny)

    # A half-span is a neutral initial Rayleigh-range estimate. When the data
    # show appreciable growth, also estimate zR from the farthest point.
    z_rayleigh_guess = max(0.5 * z_span, np.finfo(float).eps)
    farthest_index = int(np.argmax(np.abs(z - z0_guess)))
    dz_far = abs(float(z[farthest_index] - z0_guess))
    ratio_squared = (float(w[farthest_index]) / w0_guess) ** 2 - 1.0
    if dz_far > 0 and ratio_squared > 1e-6:
        z_rayleigh_guess = dz_far / math.sqrt(ratio_squared)

    positive_steps = np.diff(np.unique(z))
    min_step = float(np.min(positive_steps[positive_steps > 0]))
    z_rayleigh_min = max(min_step * 1e-4, z_span * 1e-9, 1e-12)
    z_rayleigh_max = max(z_span * 1e6, z_rayleigh_min * 10.0)
    w0_min = max(float(np.min(w)) * 1e-6, 1e-12)
    w0_max = max(float(np.max(w)) * 10.0, w0_min * 10.0)
    z0_margin = 2.0 * z_span

    p0 = np.array(
        [math.log(w0_guess), z0_guess, math.log(z_rayleigh_guess)], dtype=float
    )
    lower = np.array(
        [math.log(w0_min), float(np.min(z) - z0_margin), math.log(z_rayleigh_min)]
    )
    upper = np.array(
        [math.log(w0_max), float(np.max(z) + z0_margin), math.log(z_rayleigh_max)]
    )
    p0 = np.maximum(p0, lower + 1e-12)
    p0 = np.minimum(p0, upper - 1e-12)

    def unpack(params: np.ndarray) -> tuple[float, float, float]:
        return math.exp(float(params[0])), float(params[1]), math.exp(float(params[2]))

    def residuals(params: np.ndarray) -> np.ndarray:
        w0, z0, z_rayleigh = unpack(params)
        model = gaussian_beam_radius(z, w0, z0, z_rayleigh)
        return (model - w) / sigma

    try:
        result = least_squares(
            residuals,
            p0,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            loss="soft_l1" if robust_loss else "linear",
            f_scale=1.0,
            max_nfev=10000,
        )
    except Exception as exc:
        return failed_propagation_fit(f"{type(exc).__name__}: {exc}", z.size)

    w0, z0, z_rayleigh = unpack(result.x)
    model = gaussian_beam_radius(z, w0, z0, z_rayleigh)
    raw_residuals = w - model
    ss_res = float(np.sum(raw_residuals**2))
    ss_tot = float(np.sum((w - np.mean(w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = math.sqrt(ss_res / z.size)
    dof = max(z.size - 3, 1)
    reduced_chi2 = float(np.sum(result.fun**2) / dof)

    covariance_physical = np.full((3, 3), np.nan)
    errors = np.full(3, np.nan)
    if result.jac.size > 0:
        try:
            covariance_transformed = np.linalg.pinv(result.jac.T @ result.jac)
            # The per-image radius errors are absolute 1-sigma uncertainties,
            # so their magnitude is already present in the normalized Jacobian.
            # Do not shrink below that scale, but inflate for excess scatter.
            # The unweighted fallback always needs the residual-variance scale.
            covariance_scale = (
                max(reduced_chi2, 1.0) if uses_measured_errors else reduced_chi2
            )
            covariance_transformed *= covariance_scale
            transform = np.diag([w0, 1.0, z_rayleigh])
            covariance_physical = (
                transform @ covariance_transformed @ transform.T
            )
            diagonal = np.diag(covariance_physical)
            errors = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))
        except np.linalg.LinAlgError:
            pass

    success = bool(
        result.success
        and np.all(np.isfinite([w0, z0, z_rayleigh]))
        and w0 > 0
        and z_rayleigh > 0
    )
    message = result.message
    if not uses_measured_errors:
        message = f"{message} (unweighted: usable waist errors unavailable)"

    return PropagationFit(
        success=success,
        message=message,
        w0=w0,
        z0=z0,
        z_rayleigh=z_rayleigh,
        w0_error=float(errors[0]),
        z0_error=float(errors[1]),
        z_rayleigh_error=float(errors[2]),
        covariance=covariance_physical,
        r2=r2,
        rmse=rmse,
        reduced_chi2=reduced_chi2,
        n_points=z.size,
    )


def propagation_fit_label(
    axis_label: str,
    fit: PropagationFit,
    waist_unit: str,
    distance_unit: str,
) -> str:
    if not fit.success:
        return f"{axis_label} hyperbola fit failed"
    return (
        f"{axis_label} fit: w₀={fit.w0:.4g}±{fit.w0_error:.2g} {waist_unit}, "
        f"z₀={fit.z0:.5g}±{fit.z0_error:.2g} {distance_unit}, "
        f"zR={fit.z_rayleigh:.4g}±{fit.z_rayleigh_error:.2g} {distance_unit}"
    )


def write_propagation_csv(
    rows: Sequence[tuple[str, str, PropagationFit]],
    output_path: Path,
    waist_unit: str,
    distance_unit: str,
) -> None:
    wu = safe_column_unit(waist_unit)
    du = safe_column_unit(distance_unit)
    columns = [
        "fit_group",
        "axis",
        "success",
        "message",
        "n_points",
        f"w0_1e2_radius_{wu}",
        f"w0_error_{wu}",
        f"z0_{du}",
        f"z0_error_{du}",
        f"z_rayleigh_{du}",
        f"z_rayleigh_error_{du}",
        f"far_field_radius_slope_{wu}_per_{du}",
        "r2",
        f"rmse_{wu}",
        "reduced_chi2",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for fit_group, axis, fit in rows:
            writer.writerow(
                {
                    "fit_group": fit_group,
                    "axis": axis,
                    "success": fit.success,
                    "message": fit.message,
                    "n_points": fit.n_points,
                    f"w0_1e2_radius_{wu}": fit.w0,
                    f"w0_error_{wu}": fit.w0_error,
                    f"z0_{du}": fit.z0,
                    f"z0_error_{du}": fit.z0_error,
                    f"z_rayleigh_{du}": fit.z_rayleigh,
                    f"z_rayleigh_error_{du}": fit.z_rayleigh_error,
                    f"far_field_radius_slope_{wu}_per_{du}": fit.far_field_slope,
                    "r2": fit.r2,
                    f"rmse_{wu}": fit.rmse,
                    "reduced_chi2": fit.reduced_chi2,
                }
            )


def plot_waists(
    records: Sequence[ImageRecord],
    fit_selector,
    label_1: str,
    label_2: str,
    title: str,
    output_path: Path,
    pixel_size: float,
    waist_unit: str,
    distance_unit: str,
    robust_loss: bool,
) -> tuple[PropagationFit, PropagationFit]:
    distance, w1, e1, w2, e2 = finite_plot_arrays(
        records, fit_selector=fit_selector, pixel_size=pixel_size
    )
    if distance.size == 0:
        failed = failed_propagation_fit("No successful image fits available.", 0)
        return failed, failed

    order = np.argsort(distance)
    distance, w1, e1, w2, e2 = (
        array[order] for array in (distance, w1, e1, w2, e2)
    )

    propagation_1 = fit_gaussian_beam_propagation(
        distance, w1, e1, robust_loss=robust_loss
    )
    propagation_2 = fit_gaussian_beam_propagation(
        distance, w2, e2, robust_loss=robust_loss
    )

    fig, ax = plt.subplots(figsize=(10.4, 6.2), constrained_layout=True)
    ax.errorbar(
        distance,
        w1,
        yerr=e1,
        marker="o",
        linestyle="none",
        capsize=3,
        label=f"{label_1} data (1σ image-fit error)",
    )
    ax.errorbar(
        distance,
        w2,
        yerr=e2,
        marker="o",
        linestyle="none",
        capsize=3,
        label=f"{label_2} data (1σ image-fit error)",
    )

    z_dense = np.linspace(float(np.min(distance)), float(np.max(distance)), 800)
    if propagation_1.success:
        fitted_radius_1 = gaussian_beam_radius(
            z_dense,
            propagation_1.w0,
            propagation_1.z0,
            propagation_1.z_rayleigh,
        )
        fitted_error_1 = gaussian_beam_radius_standard_error(
            z_dense,
            propagation_1.w0,
            propagation_1.z0,
            propagation_1.z_rayleigh,
            propagation_1.covariance,
        )
        (line_1,) = ax.plot(
            z_dense,
            fitted_radius_1,
            linewidth=1.7,
            label=propagation_fit_label(
                label_1, propagation_1, waist_unit, distance_unit
            ),
        )
        ax.fill_between(
            z_dense,
            fitted_radius_1 - fitted_error_1,
            fitted_radius_1 + fitted_error_1,
            color=line_1.get_color(),
            alpha=0.18,
            linewidth=0,
        )
    if propagation_2.success:
        fitted_radius_2 = gaussian_beam_radius(
            z_dense,
            propagation_2.w0,
            propagation_2.z0,
            propagation_2.z_rayleigh,
        )
        fitted_error_2 = gaussian_beam_radius_standard_error(
            z_dense,
            propagation_2.w0,
            propagation_2.z0,
            propagation_2.z_rayleigh,
            propagation_2.covariance,
        )
        (line_2,) = ax.plot(
            z_dense,
            fitted_radius_2,
            linewidth=1.7,
            label=propagation_fit_label(
                label_2, propagation_2, waist_unit, distance_unit
            ),
        )
        ax.fill_between(
            z_dense,
            fitted_radius_2 - fitted_error_2,
            fitted_radius_2 + fitted_error_2,
            color=line_2.get_color(),
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel(f"Distance from TIFF filename [{distance_unit}]")
    ax.set_ylabel(f"Beam waist radius w (1/e² intensity) [{waist_unit}]")
    ax.set_title(
        title
        + "\n"
        + r"Gaussian-beam fit: $w(z)=w_0\sqrt{1+((z-z_0)/z_R)^2}$"
        + "\nError bars are 1σ image-fit errors; shading is the propagated 1σ fit error"
    )
    # ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return propagation_1, propagation_2


def build_parser() -> argparse.ArgumentParser:
    defaults = BeamProfileSettings(folder=Path("."))
    parser = argparse.ArgumentParser(
        description=(
            "Fit numeric-named beam TIFFs with aligned, free-angle, and "
            "common-principal-axis 2-D Gaussians."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    parser.add_argument("folder", type=Path, help="Folder containing numeric TIFFs")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder (default: FOLDER/beam_profile_results)",
    )
    parser.add_argument(
        "--centre-x",
        "--center-x",
        type=float,
        default=defaults.centre_x,
        help="Approximate beam x coordinate / image column (default: 671)",
    )
    parser.add_argument(
        "--centre-y",
        "--center-y",
        type=float,
        default=defaults.centre_y,
        help="Approximate beam y coordinate / image row (default: 786)",
    )
    parser.add_argument(
        "--fit-half-size",
        type=int,
        default=defaults.fit_half_size,
        help="Half-width of the square first-pass fit crop in pixels (default: 180)",
    )
    parser.add_argument(
        "--fit-stride",
        type=int,
        default=defaults.fit_stride,
        help="Use every Nth pixel in each direction during fitting (default: 1)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=defaults.pixel_size,
        help=(
            "Object-plane size represented by one TIFF array pixel, including "
            "camera binning and imaging magnification. Default 1, so reported "
            "waists are in pixels."
        ),
    )
    parser.add_argument(
        "--waist-unit",
        default=defaults.waist_unit,
        help="Unit label after applying --pixel-size (default: pixel)",
    )
    parser.add_argument(
        "--distance-unit",
        default=defaults.distance_unit,
        help="Unit label for numeric TIFF stems (default: filename unit)",
    )
    parser.add_argument(
        "--min-angle-r2",
        type=float,
        default=defaults.min_angle_r2,
        help=(
            "R² below which a free fit contributes no common-angle weight; "
            "weight rises smoothly above this value (default: 0.80)"
        ),
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        default=defaults.robust,
        help=(
            "Use a soft-L1 loss to reduce hot-pixel/outlier influence. "
            "Reported covariance errors are then approximate."
        ),
    )
    return parser


def analyze_beam_profiles(settings: BeamProfileSettings) -> BeamProfileResult:
    """Run the beam-profile tool from Python code.

    The analysis writes CSV files and plots to the configured output folder and
    returns their paths together with the fit summaries.
    """
    print(f"Beam-profile script version: {SCRIPT_VERSION}")
    folder = Path(settings.folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"not a directory: {folder}")
    if settings.fit_half_size < 8:
        raise ValueError("--fit-half-size must be at least 8.")
    if settings.fit_stride < 1:
        raise ValueError("--fit-stride must be at least 1.")
    if settings.pixel_size <= 0:
        raise ValueError("--pixel-size must be positive.")

    output = (
        Path(settings.output).expanduser().resolve()
        if settings.output is not None
        else folder / "beam_profile_results"
    )
    diagnostics = output / "per_image"
    diagnostics.mkdir(parents=True, exist_ok=True)

    paths = find_tiffs(folder)
    if not paths:
        raise ValueError(f"no numeric-named .tif/.tiff files found in {folder}")

    print("Gaussian convention:")
    print("  I = background + amplitude * exp[-2((u/w_u)^2 + (v/w_v)^2)]")
    print("  All reported w values are 1/e^2 intensity radii; w = 2 sigma.\n")
    print(
        f"Found {len(paths)} image(s). First-pass centre: "
        f"({settings.centre_x}, {settings.centre_y})"
    )

    records: list[ImageRecord] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] First-pass fits: {path.name}")
        try:
            image = read_tiff_2d(path)
            aligned = fit_image(
                image,
                center_guess=(settings.centre_x, settings.centre_y),
                half_size=settings.fit_half_size,
                theta_fixed=0.0,
                fit_stride=settings.fit_stride,
                robust_loss=settings.robust,
            )
            free = fit_image(
                image,
                center_guess=(settings.centre_x, settings.centre_y),
                half_size=settings.fit_half_size,
                theta_fixed=None,
                fit_stride=settings.fit_stride,
                robust_loss=settings.robust,
            )
            records.append(
                ImageRecord(
                    path=path,
                    distance=numeric_stem(path),
                    aligned=aligned,
                    free=free,
                )
            )
            if not aligned.success:
                print(f"  Warning: aligned fit failed: {aligned.message}", file=sys.stderr)
            if not free.success:
                print(f"  Warning: free fit failed: {free.message}", file=sys.stderr)
        except Exception as exc:
            print(f"  Error reading/fitting {path.name}: {exc}; skipping.", file=sys.stderr)

    if not records:
        raise RuntimeError("no images could be fitted.")

    common_theta, coherence, effective_n = average_axis_set(
        records, min_r2=settings.min_angle_r2
    )

    print("\nCommon orthogonal principal-axis set:")
    print(f"  u-axis angle from +x = {np.rad2deg(common_theta):.6f} deg")
    print(f"  v-axis angle from +x = {np.rad2deg(common_theta + np.pi/2):.6f} deg")
    print(f"  4-theta weighted coherence = {coherence:.6f}")
    print(f"  effective number of weighted images = {effective_n:.3f}")
    print(
        "  Weighting: ellipticity^2 × fit-quality^2 × SNR-quality^2 "
        "÷ angular-variance."
    )

    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] Common-axis refit and plot: {record.path.name}")
        image = read_tiff_2d(record.path)
        if record.free.success:
            center_guess = (record.free.x0, record.free.y0)
        elif record.aligned.success:
            center_guess = (record.aligned.x0, record.aligned.y0)
        else:
            center_guess = (settings.centre_x, settings.centre_y)

        record.common = fit_image(
            image,
            center_guess=center_guess,
            half_size=settings.fit_half_size,
            theta_fixed=common_theta,
            fit_stride=settings.fit_stride,
            robust_loss=settings.robust,
        )
        if not record.common.success:
            print(
                f"  Warning: common-axis fit failed: {record.common.message}",
                file=sys.stderr,
            )
            continue

        plot_name = f"{record.path.stem}_beam_profile.png"
        diagnostic_plot(
            image,
            record,
            output_path=diagnostics / plot_name,
            pixel_size=settings.pixel_size,
            waist_unit=settings.waist_unit,
            distance_unit=settings.distance_unit,
            common_theta=common_theta,
        )

    csv_path = output / "beam_fit_results.csv"
    write_csv(
        records,
        csv_path,
        common_theta=common_theta,
        pixel_size=settings.pixel_size,
        waist_unit=settings.waist_unit,
        distance_unit=settings.distance_unit,
    )

    centre_plot_path = output / "beam_centres.png"
    plot_centres(
        records,
        output_path=centre_plot_path,
        distance_unit=settings.distance_unit,
    )

    waists_xy_plot_path = output / "waists_xy.png"
    aligned_propagation = plot_waists(
        records,
        fit_selector=lambda record: record.aligned,
        label_1="x-axis radius w_x",
        label_2="y-axis radius w_y",
        title="x/y-aligned Gaussian beam radii",
        output_path=waists_xy_plot_path,
        pixel_size=settings.pixel_size,
        waist_unit=settings.waist_unit,
        distance_unit=settings.distance_unit,
        robust_loss=settings.robust,
    )

    theta_deg = np.rad2deg(common_theta)
    waists_principal_axes_plot_path = output / "waists_principal_axes.png"
    common_propagation = plot_waists(
        records,
        fit_selector=lambda record: record.common,
        label_1=f"common u-axis radius (theta={theta_deg:.3f} deg)",
        label_2=f"common v-axis radius (theta={theta_deg + 90.0:.3f} deg)",
        title="Common-principal-axis Gaussian beam radii",
        output_path=waists_principal_axes_plot_path,
        pixel_size=settings.pixel_size,
        waist_unit=settings.waist_unit,
        distance_unit=settings.distance_unit,
        robust_loss=settings.robust,
    )

    propagation_csv_path = output / "beam_propagation_fits.csv"
    propagation_rows = [
        ("x_y_aligned", "x", aligned_propagation[0]),
        ("x_y_aligned", "y", aligned_propagation[1]),
        ("common_principal_axes", "u", common_propagation[0]),
        ("common_principal_axes", "v", common_propagation[1]),
    ]
    write_propagation_csv(
        propagation_rows,
        propagation_csv_path,
        waist_unit=settings.waist_unit,
        distance_unit=settings.distance_unit,
    )

    print("\nGaussian-beam propagation fits:")
    print("  w(z) = w0 * sqrt(1 + ((z - z0) / zR)^2)")
    print("  w0 is a 1/e^2 intensity radius.")
    for fit_group, axis, fit in propagation_rows:
        if fit.success:
            print(
                f"  {fit_group}/{axis}: "
                f"w0={fit.w0:.6g} ± {fit.w0_error:.3g} {settings.waist_unit}, "
                f"z0={fit.z0:.6g} ± {fit.z0_error:.3g} {settings.distance_unit}, "
                f"zR={fit.z_rayleigh:.6g} ± {fit.z_rayleigh_error:.3g} "
                f"{settings.distance_unit}, R²={fit.r2:.6f}"
            )
        else:
            print(f"  {fit_group}/{axis}: fit failed: {fit.message}")

    successful_common = sum(
        record.common is not None and record.common.success for record in records
    )
    print("\nFinished.")
    print(f"  Successful common-axis fits: {successful_common}/{len(records)}")
    print(f"  Results CSV: {csv_path}")
    print(f"  Propagation-fit CSV: {propagation_csv_path}")
    print(f"  Per-image figures: {diagnostics}")
    print(f"  beam-centre plot: {centre_plot_path}")
    print(f"  x/y waist plot: {waists_xy_plot_path}")
    print(f"  principal-axis waist plot: {waists_principal_axes_plot_path}")
    return BeamProfileResult(
        records=tuple(records),
        output=output,
        diagnostics=diagnostics,
        results_csv=csv_path,
        propagation_csv=propagation_csv_path,
        centre_plot=centre_plot_path,
        waists_xy_plot=waists_xy_plot_path,
        waists_principal_axes_plot=waists_principal_axes_plot_path,
        common_theta=common_theta,
        common_angle_coherence=coherence,
        common_angle_effective_n=effective_n,
        aligned_propagation=aligned_propagation,
        common_propagation=common_propagation,
    )


def _settings_from_args(args: argparse.Namespace) -> BeamProfileSettings:
    return BeamProfileSettings(
        folder=args.folder,
        output=args.output,
        centre_x=args.centre_x,
        centre_y=args.centre_y,
        fit_half_size=args.fit_half_size,
        fit_stride=args.fit_stride,
        pixel_size=args.pixel_size,
        waist_unit=args.waist_unit,
        distance_unit=args.distance_unit,
        min_angle_r2=args.min_angle_r2,
        robust=args.robust,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for ``dnamic-beam-profile``."""
    try:
        analyze_beam_profiles(_settings_from_args(build_parser().parse_args(argv)))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
