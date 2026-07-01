"""Utility functions for the PyTorch ECINN example.

This file mirrors the TensorFlow helper.py behavior for the irreversible
Fe(III) reduction ECINN case.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import interpolate

# Physical constants used by the original TensorFlow example.
R = 8.314
T = 298.0
F = 96485.0


def to_dimensional_potential(theta: np.ndarray | float) -> np.ndarray | float:
    """Convert dimensionless potential theta to V relative to E_ref."""
    return theta / (F / (R * T))


def to_dimensionless_potential(volts: np.ndarray | float) -> np.ndarray | float:
    """Convert V relative to E_ref to dimensionless potential theta."""
    return volts * (F / (R * T))


def _as_numpy_1d(x: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr.reshape(-1)


def flux_sampling(time_array: np.ndarray, df_fd: pd.DataFrame, max_t: float) -> np.ndarray:
    """Interpolate a simulated finite-difference flux onto requested times.

    This is kept for compatibility with the original helper.py.
    """
    time_array = _as_numpy_1d(time_array)
    interpolated_flux = np.zeros_like(time_array, dtype=np.float64)
    n = len(time_array)

    df_forward = df_fd.iloc[: int(len(df_fd) / 2)]
    df_reverse = df_fd.iloc[int(len(df_fd) / 2) :]

    forward_mask = np.arange(n) < int(n / 2)
    forward_times = time_array[forward_mask]
    reverse_times = time_array[~forward_mask]

    f_forward = interpolate.interp1d(
        np.linspace(0, max_t / 2.0, num=len(df_forward)),
        df_forward.iloc[:, 1].to_numpy(dtype=np.float64),
        bounds_error=False,
        fill_value="extrapolate",
    )
    f_reverse = interpolate.interp1d(
        np.linspace(max_t / 2.0, max_t, num=len(df_reverse)),
        df_reverse.iloc[:, 1].to_numpy(dtype=np.float64),
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_flux[forward_mask] = f_forward(forward_times)
    interpolated_flux[~forward_mask] = f_reverse(reverse_times)
    return interpolated_flux


def exp_flux_sampling(
    time_array: np.ndarray,
    df_exp: pd.DataFrame,
    full_scan_t: float,
    portion_analyzed: float = 0.75,
) -> np.ndarray:
    """Interpolate experimental dimensionless flux at given dimensionless times.

    The original implementation splits the voltammogram into forward and reverse
    halves and only uses the first ``portion_analyzed`` fraction of the full scan.
    This implementation accepts sorted or unsorted time arrays.
    """
    time_array = _as_numpy_1d(time_array)
    out = np.zeros_like(time_array, dtype=np.float64)

    df_forward = df_exp.iloc[: int(len(df_exp) * 0.5)]
    df_reverse = df_exp.iloc[int(len(df_exp) * 0.5) : int(len(df_exp) * portion_analyzed)]

    if len(df_forward) < 2 or len(df_reverse) < 2:
        raise ValueError(
            "Experimental dataframe is too short after forward/reverse split. "
            "Check portion_analyzed and input file."
        )

    f_forward = interpolate.interp1d(
        np.linspace(0, full_scan_t * 0.5, num=len(df_forward)),
        df_forward.iloc[:, 1].to_numpy(dtype=np.float64),
        bounds_error=False,
        fill_value="extrapolate",
    )
    f_reverse = interpolate.interp1d(
        np.linspace(full_scan_t * 0.5, full_scan_t * portion_analyzed, num=len(df_reverse)),
        df_reverse.iloc[:, 1].to_numpy(dtype=np.float64),
        bounds_error=False,
        fill_value="extrapolate",
    )

    forward = time_array < full_scan_t * 0.5
    out[forward] = f_forward(time_array[forward])
    out[~forward] = f_reverse(time_array[~forward])
    return out


def find_csv(path_to_dir: str | Path = ".", suffix: str = ".txt") -> list[str]:
    path = Path(path_to_dir)
    return [
        p.name
        for p in path.iterdir()
        if p.name.endswith(suffix)
        and "Experimental" not in p.name
        and "One Electron Reduction" not in p.name
    ]


def find_experimental_csv(
    path_to_dir: str | Path = ".", prefix: str = "Experimental", suffix: str = ".csv"
) -> list[str]:
    path = Path(path_to_dir)
    return [p.name for p in path.iterdir() if p.name.endswith(suffix) and prefix in p.name]


def exp_parameters(file_name: str | Path) -> tuple[float, float, float, float]:
    """Parse sigma, theta_i, theta_v, and dA from an Exp Dimensionless filename."""
    text = Path(file_name).name.replace(".csv", "")

    def grab(pattern: str, name: str) -> float:
        m = re.findall(pattern, text)
        if not m:
            raise ValueError(f"Could not parse {name} from filename: {file_name}")
        return float(m[0])

    sigma = grab(r"sigma=([+-]?[\d.]+[eE][+-][\d]+)", "sigma")
    theta_i = grab(r"theta_i=([+-]?[\d.]+[eE][+-][\d]+)", "theta_i")
    theta_v = grab(r"theta_v=([+-]?[\d.]+[eE][+-][\d]+)", "theta_v")
    d_a = grab(r"dA=([+-]?[\d.]+[eE][+-][\d]+)", "dA")
    return sigma, theta_i, theta_v, d_a


def find_sigma(cv: str | Path) -> tuple[float, float, float, float, float, float]:
    text = Path(cv).name.replace(".csv", "")
    vals = []
    for i in range(1, 7):
        m = re.findall(rf"var{i}=([+-]?[\d.]+[eE][+-][\d]+)", text)
        if not m:
            raise ValueError(f"Could not parse var{i} from filename: {cv}")
        vals.append(float(m[0]))
    return tuple(vals)  # type: ignore[return-value]


def find_conc(cv: str | Path):
    text = Path(cv).name.replace(".csv", "")
    point_m = re.findall(r"Point=([A-Z])", text)
    theta_m = re.findall(r"Theta=([-]?[\d.]+[eE][+-][\d]+)", text)
    if not point_m or not theta_m:
        raise ValueError(f"Could not parse Point/Theta from filename: {cv}")
    vars_ = find_sigma(cv)
    return point_m[0], float(theta_m[0]), *vars_


def find_point(cvs: Iterable[str], point: str) -> list[str]:
    pattern = re.compile(f"Point={point}.*")
    match = []
    for cv in cvs:
        m = pattern.findall(cv)
        if m:
            match.append(m[0])
    return match


def format_func_dimensional_potential(value: float, tick_number: int | None = None) -> str:
    """Convert dimensionless potential to mV string for matplotlib tick labels."""
    value_mv = value / F * R * T * 1e3
    return f"{value_mv:.2f}"
