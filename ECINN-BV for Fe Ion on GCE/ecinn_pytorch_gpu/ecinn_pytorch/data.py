"""GPU-friendly on-the-fly collocation sampler for ECINN training.

This version avoids the main CPU bottleneck in the first PyTorch draft:
NumPy/Pandas/SciPy interpolation inside every training step. Experimental
flux curves are preloaded as torch tensors on the selected device, and each
batch is generated directly with torch.rand on that same device.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch

from .helper import exp_parameters

OuterBoundary = Literal["SI", "TL"]


@dataclass
class ExperimentData:
    file_name: str
    df_exp: pd.DataFrame
    sigma: float
    theta_i: float
    theta_v: float
    full_scan_t: float
    max_t: float
    max_x: float
    # Precomputed interpolation nodes on the training device.
    t_forward: torch.Tensor
    j_forward: torch.Tensor
    t_reverse: torch.Tensor
    j_reverse: torch.Tensor


def torch_interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear 1D interpolation fully in torch.

    Parameters
    ----------
    x:
        Query points, arbitrary shape.
    xp:
        1D sorted sample locations.
    fp:
        1D sample values corresponding to xp.

    Returns
    -------
    torch.Tensor
        Interpolated values with the same shape as x. Query points outside
        the tabulated range are clamped to the end intervals, matching a safe
        extrapolation behavior for the small boundary overshoots that may occur.
    """
    x_flat = x.reshape(-1)
    idx = torch.searchsorted(xp, x_flat, right=False)
    idx = torch.clamp(idx, 1, xp.numel() - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    weight = (x_flat - x0) / torch.clamp(x1 - x0, min=torch.finfo(x.dtype).eps)
    y = y0 + weight * (y1 - y0)
    return y.reshape_as(x)


class ECINNSampler:
    """Generate ECINN collocation, boundary, and experiment-fitting batches.

    This class is device-native: when ``device='cuda'``, random points,
    potential values, and interpolated experimental fluxes are created on GPU.
    That substantially reduces CPU-GPU transfer overhead.
    """

    def __init__(
        self,
        exp_file_names: list[str | Path],
        sigmas: list[float] | None = None,
        theta_i: float | None = None,
        theta_v: float | None = None,
        portion_analyzed: float = 0.75,
        lambda_x: float = 3.0,
        batch_size: int = 250,
        device: str | torch.device = "cpu",
    ) -> None:
        if len(exp_file_names) < 1:
            raise ValueError("At least one experimental file is required")

        self.exp_file_names = [str(p) for p in exp_file_names]
        self.portion_analyzed = float(portion_analyzed)
        self.lambda_x = float(lambda_x)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = torch.float32

        parsed = [exp_parameters(p) for p in self.exp_file_names]
        parsed_sigmas = [p[0] for p in parsed]
        parsed_theta_i = parsed[0][1]
        parsed_theta_v = parsed[0][2]

        if sigmas is None:
            sigmas = parsed_sigmas
        if theta_i is None:
            theta_i = parsed_theta_i
        if theta_v is None:
            theta_v = parsed_theta_v

        self.sigmas = [float(s) for s in sigmas]
        self.theta_i = float(theta_i)
        self.theta_v = float(theta_v)

        if len(self.sigmas) != len(self.exp_file_names):
            raise ValueError("sigmas length must match exp_file_names length")

        self.experiments: list[ExperimentData] = []
        for file_name, sigma in zip(self.exp_file_names, self.sigmas):
            df = pd.read_csv(file_name)
            full_scan_t = 2.0 * abs(self.theta_v - self.theta_i) / sigma
            max_t = full_scan_t * self.portion_analyzed
            max_x = self.lambda_x * (max_t ** 0.5)

            n_f = int(len(df) * 0.5)
            n_r0 = int(len(df) * 0.5)
            n_r1 = int(len(df) * self.portion_analyzed)
            if n_f < 2 or (n_r1 - n_r0) < 2:
                raise ValueError(
                    "Experimental dataframe is too short after forward/reverse split. "
                    "Check portion_analyzed and input file."
                )

            # The original helper assumes a uniformly spaced time axis for each
            # branch of the voltammogram. We keep the same convention.
            t_forward = torch.linspace(
                0.0, full_scan_t * 0.5, steps=n_f, device=self.device, dtype=self.dtype
            )
            j_forward = torch.as_tensor(
                df.iloc[:n_f, 1].to_numpy(), device=self.device, dtype=self.dtype
            )
            t_reverse = torch.linspace(
                full_scan_t * 0.5,
                full_scan_t * self.portion_analyzed,
                steps=n_r1 - n_r0,
                device=self.device,
                dtype=self.dtype,
            )
            j_reverse = torch.as_tensor(
                df.iloc[n_r0:n_r1, 1].to_numpy(), device=self.device, dtype=self.dtype
            )

            self.experiments.append(
                ExperimentData(
                    file_name=str(file_name),
                    df_exp=df,
                    sigma=float(sigma),
                    theta_i=self.theta_i,
                    theta_v=self.theta_v,
                    full_scan_t=float(full_scan_t),
                    max_t=float(max_t),
                    max_x=float(max_x),
                    t_forward=t_forward,
                    j_forward=j_forward,
                    t_reverse=t_reverse,
                    j_reverse=j_reverse,
                )
            )

    @property
    def num_experiments(self) -> int:
        return len(self.experiments)

    def _potential(self, t: torch.Tensor, exp: ExperimentData) -> torch.Tensor:
        half_t = exp.full_scan_t / 2.0
        return torch.where(
            t < half_t,
            exp.theta_i - exp.sigma * t,
            exp.theta_v + exp.sigma * (t - half_t),
        )

    def _flux_exp(self, t: torch.Tensor, exp: ExperimentData) -> torch.Tensor:
        half_t = exp.full_scan_t / 2.0
        forward = t < half_t
        out = torch.empty_like(t)
        if torch.any(forward):
            out[forward] = torch_interp1d(t[forward], exp.t_forward, exp.j_forward)
        if torch.any(~forward):
            out[~forward] = torch_interp1d(t[~forward], exp.t_reverse, exp.j_reverse)
        return out

    def sample_one(self, exp: ExperimentData) -> dict[str, torch.Tensor]:
        b = self.batch_size
        dev = self.device
        dtype = self.dtype

        # PDE domain points: t in [0,maxT], x in [0,maxX].
        tx_eqn = torch.rand((b, 2), device=dev, dtype=dtype)
        tx_eqn[:, 0].mul_(exp.max_t)
        tx_eqn[:, 1].mul_(exp.max_x)
        tx_eqn.requires_grad_(True)

        # Initial condition: t=0, c=1.
        tx_ini = torch.rand((b, 2), device=dev, dtype=dtype)
        tx_ini[:, 0].zero_()
        tx_ini[:, 1].mul_(exp.max_x)

        # Electrode boundary: x=0, match BV and experimental flux.
        t_bnd0 = torch.rand((b,), device=dev, dtype=dtype) * exp.max_t
        # Sorting is not physically necessary, but helps match the original TF behavior.
        t_bnd0, _ = torch.sort(t_bnd0)
        tx_bnd0 = torch.zeros((b, 2), device=dev, dtype=dtype)
        tx_bnd0[:, 0] = t_bnd0
        tx_bnd0.requires_grad_(True)

        theta = self._potential(t_bnd0, exp).reshape(-1, 1)
        flux_exp = self._flux_exp(t_bnd0, exp).reshape(-1, 1)

        # Outer boundary: x=maxX.
        tx_bnd1 = torch.rand((b, 2), device=dev, dtype=dtype)
        tx_bnd1[:, 0].mul_(exp.max_t)
        tx_bnd1[:, 1].fill_(exp.max_x)
        tx_bnd1.requires_grad_(True)

        return {
            "tx_eqn": tx_eqn,
            "tx_ini": tx_ini,
            "tx_bnd0": tx_bnd0,
            "theta": theta,
            "flux_exp": flux_exp,
            "tx_bnd1": tx_bnd1,
        }

    def sample_batch(self) -> list[dict[str, torch.Tensor]]:
        return [self.sample_one(exp) for exp in self.experiments]

    def max_times(self) -> list[float]:
        return [e.max_t for e in self.experiments]

    def full_scan_times(self) -> list[float]:
        return [e.full_scan_t for e in self.experiments]

    def max_xs(self) -> list[float]:
        return [e.max_x for e in self.experiments]

    def dataframes(self) -> list[pd.DataFrame]:
        return [e.df_exp.copy() for e in self.experiments]
