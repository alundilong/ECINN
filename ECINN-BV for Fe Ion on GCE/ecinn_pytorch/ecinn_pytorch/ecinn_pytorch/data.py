"""On-the-fly collocation sampler for ECINN training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from .helper import exp_flux_sampling, exp_parameters

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


class ECINNSampler:
    """Generate PINN collocation, boundary, and experiment-fitting batches.

    This replaces the TensorFlow ``DataGenerator`` with explicit random sampling.
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
            max_x = self.lambda_x * np.sqrt(max_t)
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
                )
            )

    @property
    def num_experiments(self) -> int:
        return len(self.experiments)

    def _to_tensor(self, arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        if requires_grad:
            t.requires_grad_(True)
        return t

    def _potential(self, t: np.ndarray, exp: ExperimentData) -> np.ndarray:
        return np.where(
            t < exp.full_scan_t / 2.0,
            exp.theta_i - exp.sigma * t,
            exp.theta_v + exp.sigma * (t - exp.full_scan_t / 2.0),
        )

    def sample_one(self, exp: ExperimentData) -> dict[str, torch.Tensor]:
        b = self.batch_size

        # PDE domain points: t in [0,maxT], x in [0,maxX]
        tx_eqn = np.random.rand(b, 2)
        tx_eqn[:, 0] *= exp.max_t
        tx_eqn[:, 1] *= exp.max_x

        # Initial condition: t=0, c=1
        tx_ini = np.random.rand(b, 2)
        tx_ini[:, 0] = 0.0
        tx_ini[:, 1] *= exp.max_x

        # Electrode boundary: x=0, match BV and experimental flux.
        # Use sorted times to make interpolation/plotting behavior close to the TF code.
        t_bnd0 = np.sort(np.random.rand(b) * exp.max_t)
        tx_bnd0 = np.zeros((b, 2), dtype=np.float64)
        tx_bnd0[:, 0] = t_bnd0
        tx_bnd0[:, 1] = 0.0
        theta = self._potential(t_bnd0, exp).reshape(-1, 1)
        flux_exp = exp_flux_sampling(
            t_bnd0, exp.df_exp, exp.full_scan_t, self.portion_analyzed
        ).reshape(-1, 1)

        # Outer boundary: x=maxX
        tx_bnd1 = np.random.rand(b, 2)
        tx_bnd1[:, 0] *= exp.max_t
        tx_bnd1[:, 1] = exp.max_x

        return {
            "tx_eqn": self._to_tensor(tx_eqn, requires_grad=True),
            "tx_ini": self._to_tensor(tx_ini, requires_grad=False),
            "tx_bnd0": self._to_tensor(tx_bnd0, requires_grad=True),
            "theta": self._to_tensor(theta, requires_grad=False),
            "flux_exp": self._to_tensor(flux_exp, requires_grad=False),
            "tx_bnd1": self._to_tensor(tx_bnd1, requires_grad=True),
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
