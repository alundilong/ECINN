"""PyTorch ECINN model for irreversible cathodic Butler--Volmer kinetics."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_softplus(x: float) -> float:
    """Return y so that softplus(y) ~= x."""
    if x <= 0:
        raise ValueError("x must be positive")
    return math.log(math.expm1(x))


class MLP(nn.Module):
    """Fully connected network c(t, x)."""

    def __init__(
        self,
        num_inputs: int = 2,
        layers: Sequence[int] = (64, 32, 32, 32, 64),
        num_outputs: int = 1,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if activation != "tanh":
            raise ValueError("This example currently supports activation='tanh'")

        dims = [num_inputs, *layers, num_outputs]
        modules: list[nn.Module] = []
        for i in range(len(dims) - 2):
            linear = nn.Linear(dims[i], dims[i + 1])
            # Close in spirit to Keras he_normal. Xavier often works better for tanh,
            # but this keeps the example near the original implementation.
            nn.init.kaiming_normal_(linear.weight, nonlinearity="linear")
            nn.init.zeros_(linear.bias)
            modules += [linear, nn.Tanh()]

        last = nn.Linear(dims[-2], dims[-1])
        nn.init.kaiming_normal_(last.weight, nonlinearity="linear")
        nn.init.zeros_(last.bias)
        modules.append(last)
        self.net = nn.Sequential(*modules)

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        return self.net(tx)


def gradients(net: nn.Module, tx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute c, dc/dt, dc/dx, and d2c/dx2 for a network c(t,x)."""
    if not tx.requires_grad:
        tx = tx.detach().clone().requires_grad_(True)

    c = net(tx)
    grad_c = torch.autograd.grad(
        c,
        tx,
        grad_outputs=torch.ones_like(c),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    dc_dt = grad_c[:, 0:1]
    dc_dx = grad_c[:, 1:2]
    grad_dc_dx = torch.autograd.grad(
        dc_dx,
        tx,
        grad_outputs=torch.ones_like(dc_dx),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    d2c_dx2 = grad_dc_dx[:, 1:2]
    return c, dc_dt, dc_dx, d2c_dx2


def first_gradients(net: nn.Module, tx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute c, dc/dt, and dc/dx for boundary losses."""
    if not tx.requires_grad:
        tx = tx.detach().clone().requires_grad_(True)
    c = net(tx)
    grad_c = torch.autograd.grad(
        c,
        tx,
        grad_outputs=torch.ones_like(c),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return c, grad_c[:, 0:1], grad_c[:, 1:2]


class ECINN(nn.Module):
    """Multi-scan ECINN with shared K0, alpha, and dimensionless diffusion dA.

    Each scan rate has its own concentration network, while the electrochemical
    parameters are shared across scans.
    """

    def __init__(
        self,
        num_experiments: int = 3,
        layers: Sequence[int] = (64, 32, 32, 32, 64),
        init_k0: float = 1.0,
        init_alpha: float = 0.4,
        init_dA: float = 0.4,
        exp_clip: float = 60.0,
    ) -> None:
        super().__init__()
        self.networks = nn.ModuleList([MLP(layers=layers) for _ in range(num_experiments)])
        self.raw_K0 = nn.Parameter(torch.tensor(inverse_softplus(init_k0), dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.tensor(inverse_softplus(init_alpha), dtype=torch.float32))
        self.raw_dA = nn.Parameter(torch.tensor(inverse_softplus(init_dA), dtype=torch.float32))
        self.exp_clip = float(exp_clip)

    @property
    def K0(self) -> torch.Tensor:
        return F.softplus(self.raw_K0)

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    @property
    def dA(self) -> torch.Tensor:
        return F.softplus(self.raw_dA)

    def parameter_values(self) -> tuple[float, float, float]:
        return float(self.K0.detach().cpu()), float(self.alpha.detach().cpu()), float(self.dA.detach().cpu())

    def cathodic_bv_flux(self, theta: torch.Tensor, c_surface: torch.Tensor) -> torch.Tensor:
        exponent = torch.clamp(-self.alpha * theta, min=-self.exp_clip, max=self.exp_clip)
        return self.K0 * torch.exp(exponent) * c_surface

    def diffusion_residual(self, net_index: int, tx_eqn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c, dc_dt, _dc_dx, d2c_dx2 = gradients(self.networks[net_index], tx_eqn)
        residual = dc_dt - self.dA * d2c_dx2
        return residual, c

    def boundary_fluxes(
        self, net_index: int, tx_bnd0: torch.Tensor, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_s, _dc_dt, dc_dx = first_gradients(self.networks[net_index], tx_bnd0)
        diffusion_flux = self.dA * dc_dx
        predicted_flux = -diffusion_flux
        bv_flux = self.cathodic_bv_flux(theta, c_s)
        bv_residual = diffusion_flux - bv_flux
        return predicted_flux, bv_residual, diffusion_flux, c_s

    def outer_residual(self, net_index: int, tx_bnd1: torch.Tensor, outer_boundary: str = "SI") -> torch.Tensor:
        if outer_boundary == "SI":
            return self.networks[net_index](tx_bnd1) - 1.0
        if outer_boundary == "TL":
            # Physically meaningful thin-layer no-flux condition dc/dx=0.
            # The original TF code's TL branch appears to output concentration;
            # SI is the main mode used in the provided example.
            _c, _dc_dt, dc_dx = first_gradients(self.networks[net_index], tx_bnd1)
            return dc_dx
        raise ValueError("outer_boundary must be 'SI' or 'TL'")
