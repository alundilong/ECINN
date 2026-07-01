"""Train a PyTorch ECINN for irreversible Fe(III) reduction CV data.

Recommended use from the project root:
    python -m ecinn_pytorch.train_ecinn --epochs 300

For a quick code smoke test using nonphysical toy data:
    python -m ecinn_pytorch.train_ecinn --make-demo-data --epochs 2 --steps-per-epoch 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .convert_to_dimensionless import A, Dref, ERef, F as FARADAY, R, T, c_bulk, r_e
from .data import ECINNSampler
from .helper import exp_parameters
from .model import ECINN, gradients


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mse(x: torch.Tensor, target: float | torch.Tensor = 0.0) -> torch.Tensor:
    if isinstance(target, torch.Tensor):
        return torch.mean((x - target) ** 2)
    return torch.mean((x - float(target)) ** 2)


def bv_weight_for_epoch(epoch: int, ramp_start: int = 20, ramp_end: int = 40) -> float:
    """Mimic the TF callback: BV loss is ramped from 0 to 1 after early pretraining."""
    if epoch <= ramp_start:
        return 0.0
    if epoch >= ramp_end:
        return 1.0
    return float(epoch - ramp_start) / float(ramp_end - ramp_start)


def compute_losses(
    model: ECINN,
    batch: list[dict[str, torch.Tensor]],
    outer_boundary: str,
) -> dict[str, torch.Tensor]:
    loss_pde = torch.zeros((), device=next(model.parameters()).device)
    loss_ic = torch.zeros_like(loss_pde)
    loss_bv = torch.zeros_like(loss_pde)
    loss_exp = torch.zeros_like(loss_pde)
    loss_outer = torch.zeros_like(loss_pde)

    for i, b in enumerate(batch):
        pde_res, _ = model.diffusion_residual(i, b["tx_eqn"])
        loss_pde = loss_pde + mse(pde_res, 0.0)

        c_ini = model.networks[i](b["tx_ini"])
        loss_ic = loss_ic + mse(c_ini, 1.0)

        pred_flux, bv_res, _diffusion_flux, _c_s = model.boundary_fluxes(
            i, b["tx_bnd0"], b["theta"]
        )
        loss_bv = loss_bv + mse(bv_res, 0.0)
        loss_exp = loss_exp + mse(pred_flux, b["flux_exp"])

        outer_res = model.outer_residual(i, b["tx_bnd1"], outer_boundary=outer_boundary)
        loss_outer = loss_outer + mse(outer_res, 0.0)

    n = len(batch)
    return {
        "pde": loss_pde / n,
        "ic": loss_ic / n,
        "bv": loss_bv / n,
        "exp": loss_exp / n,
        "outer": loss_outer / n,
    }


def train(
    model: ECINN,
    sampler: ECINNSampler,
    output_dir: Path,
    epochs: int = 300,
    steps_per_epoch: int = 4000,
    lr: float = 1e-3,
    outer_boundary: str = "SI",
    resume: str | None = None,
    weight_pde: float = 1.0,
    weight_ic: float = 1.0,
    weight_exp: float = 1.0,
    weight_outer: float = 1.0,
    ramp_start: int = 20,
    ramp_end: int = 40,
    save_every: int = 25,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        state = torch.load(resume, map_location=next(model.parameters()).device)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"Loaded checkpoint: {resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, Any]] = []

    for epoch in range(epochs):
        if epoch > 50:
            for group in optimizer.param_groups:
                group["lr"] *= 0.98

        model.train()
        bv_w = bv_weight_for_epoch(epoch, ramp_start, ramp_end)

        accum = {"pde": 0.0, "ic": 0.0, "bv": 0.0, "exp": 0.0, "outer": 0.0, "total": 0.0}
        for _step in range(steps_per_epoch):
            batch = sampler.sample_batch()
            losses = compute_losses(model, batch, outer_boundary=outer_boundary)
            total = (
                weight_pde * losses["pde"]
                + weight_ic * losses["ic"]
                + bv_w * losses["bv"]
                + weight_exp * losses["exp"]
                + weight_outer * losses["outer"]
            )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()

            for key in ["pde", "ic", "bv", "exp", "outer"]:
                accum[key] += float(losses[key].detach().cpu())
            accum["total"] += float(total.detach().cpu())

        for key in accum:
            accum[key] /= steps_per_epoch
        K0, alpha, dA = model.parameter_values()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "bv_weight": bv_w,
            **{f"loss_{k}": v for k, v in accum.items()},
            "lambda_K0": K0,
            "lambda_alpha": alpha,
            "lambda_dA": dA,
        }
        history.append(row)

        print(
            f"epoch {epoch:04d} | total={accum['total']:.3e} "
            f"pde={accum['pde']:.3e} ic={accum['ic']:.3e} bv={accum['bv']:.3e} "
            f"exp={accum['exp']:.3e} outer={accum['outer']:.3e} | "
            f"wBV={bv_w:.2f} K0={K0:.4e} alpha={alpha:.4f} dA={dA:.4f}"
        )

        if save_every > 0 and ((epoch + 1) % save_every == 0 or epoch == epochs - 1):
            ckpt_path = ckpt_dir / f"ecinn_epoch_{epoch + 1:04d}.pt"
            torch.save({"model": model.state_dict(), "history": history}, ckpt_path)

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    final_path = ckpt_dir / "ecinn_final.pt"
    torch.save({"model": model.state_dict(), "history": history}, final_path)
    print(f"Saved final checkpoint: {final_path}")
    return pd.DataFrame(history)


def predict_flux(
    model: ECINN,
    net_index: int,
    t_flat: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tx_flux = np.zeros((len(t_flat), 2), dtype=np.float32)
    tx_flux[:, 0] = t_flat.astype(np.float32)
    tx = torch.tensor(tx_flux, dtype=torch.float32, device=device, requires_grad=True)
    c = model.networks[net_index](tx)
    grad_c = torch.autograd.grad(
        c,
        tx,
        grad_outputs=torch.ones_like(c),
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]
    dc_dx = grad_c[:, 1].detach().cpu().numpy()
    return -model.dA.detach().cpu().numpy() * dc_dx


def plot_predictions(
    model: ECINN,
    sampler: ECINNSampler,
    output_dir: Path,
    scan_rates: list[float] | None = None,
    num_test_samples: int = 400,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    experiments = sampler.experiments
    if scan_rates is None or len(scan_rates) != len(experiments):
        scan_rates = [float("nan")] * len(experiments)

    K0, alpha, dA = model.parameter_values()
    lambda_k0 = K0 * Dref / r_e
    lambda_k0_eff = lambda_k0 * math.exp(alpha * ERef * FARADAY / (R * T))

    n = len(experiments)
    fig_all, axs_all = plt.subplots(figsize=(8 * n, 9), nrows=2, ncols=n, squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    for i, exp in enumerate(experiments):
        t_flat = np.linspace(0, exp.max_t, num_test_samples)
        theta_flat = np.where(
            t_flat < exp.full_scan_t / 2.0,
            exp.theta_i - exp.sigma * t_flat,
            exp.theta_v + exp.sigma * (t_flat - exp.full_scan_t / 2.0),
        )
        x_flat = np.linspace(0, exp.max_x, num_test_samples)
        t, x = np.meshgrid(t_flat, x_flat)
        tx = np.stack([t.reshape(-1), x.reshape(-1)], axis=-1).astype(np.float32)

        c_chunks = []
        with torch.no_grad():
            for start in range(0, len(tx), 65536):
                tx_tensor = torch.tensor(tx[start : start + 65536], device=device)
                c_chunks.append(model.networks[i](tx_tensor).detach().cpu().numpy())
        c = np.concatenate(c_chunks, axis=0).reshape(t.shape)
        flux = predict_flux(model, i, t_flat, device)

        df_pred = pd.DataFrame({"Potential": theta_flat, "Flux": flux})
        df_pred.to_csv(output_dir / f"PINN_scan_{i}.csv", index=False)

        # Dimensionless concentration field and CV.
        ax = axs_all[0, i]
        pc = ax.pcolormesh(t, x, c, cmap="rainbow", shading="auto")
        ax.set_xlabel("T")
        ax.set_ylabel("X")
        pc.set_clim(0.0, 1.0)
        cbar = plt.colorbar(pc, ax=ax, pad=0.05, aspect=10)
        cbar.set_label("C(T,X)")
        ax.set_title(f"scan {i}: sigma={exp.sigma:.2E}")

        ax = axs_all[1, i]
        ax.plot(theta_flat, flux, lw=2.5, label="PyTorch ECINN")
        df_exp = exp.df_exp.iloc[: int(sampler.portion_analyzed * len(exp.df_exp))]
        ax.plot(df_exp.iloc[:, 0], df_exp.iloc[:, 1], "--", lw=2.5, label="Experiment")
        ax.set_xlabel(r"Potential, $\theta$")
        ax.set_ylabel(r"Flux, $J$")
        ax.legend()

    fig_all.suptitle(
        f"lambda_K0={K0:.3E}, lambda_alpha={alpha:.3f}, lambda_dA={dA:.3f}\n"
        f"k0={lambda_k0:.3E} m/s, k0_eff={lambda_k0_eff:.3E} m/s, "
        f"D={dA * Dref:.3E} m^2/s",
        y=1.02,
    )
    fig_all.tight_layout()
    fig_all.savefig(output_dir / "PINN_dimensionless_summary.png", dpi=250, bbox_inches="tight")
    plt.close(fig_all)

    # Paper-style dimensional plot.
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(2, n, hspace=0.35)
    ax_cv = fig.add_subplot(gs[1, :])
    for i, exp in enumerate(experiments):
        t_flat = np.linspace(0, exp.max_t, num_test_samples)
        theta_flat = np.where(
            t_flat < exp.full_scan_t / 2.0,
            exp.theta_i - exp.sigma * t_flat,
            exp.theta_v + exp.sigma * (t_flat - exp.full_scan_t / 2.0),
        )
        x_flat = np.linspace(0, exp.max_x, num_test_samples)
        t, x = np.meshgrid(t_flat, x_flat)
        tx = np.stack([t.reshape(-1), x.reshape(-1)], axis=-1).astype(np.float32)
        c_chunks = []
        with torch.no_grad():
            for start in range(0, len(tx), 65536):
                tx_tensor = torch.tensor(tx[start : start + 65536], device=device)
                c_chunks.append(model.networks[i](tx_tensor).detach().cpu().numpy())
        c = np.concatenate(c_chunks, axis=0).reshape(t.shape)
        flux = predict_flux(model, i, t_flat, device)

        ax = fig.add_subplot(gs[0, i])
        t_dim = t * r_e * r_e / Dref
        x_dim = x * r_e * 1e3
        c_dim = c * c_bulk
        pc = ax.pcolormesh(t_dim, x_dim, c_dim, cmap="YlGnBu", shading="auto")
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("t / s")
        if i == 0:
            ax.set_ylabel("x / mm")
        else:
            ax.set_yticks([])
        if i == n - 1:
            cbar = plt.colorbar(pc, pad=0.15, aspect=28, ax=ax)
            cbar.set_label(r"$c_{Fe^{3+}}$ / mM")

        potential_dim = theta_flat / (FARADAY / (R * T)) + ERef
        current_dim = flux * (FARADAY * A * c_bulk * Dref / r_e) * 1e6
        label = f"scan {i}" if math.isnan(scan_rates[i]) else rf"$\nu={scan_rates[i]:.2f}$ V/s"
        ax_cv.plot(potential_dim, current_dim, lw=2.5, color=colors[i], label=label)

        df_exp = exp.df_exp.iloc[: int(sampler.portion_analyzed * len(exp.df_exp))].copy()
        exp_potential = df_exp.iloc[:, 0].to_numpy() / (FARADAY / (R * T)) + ERef
        exp_current = df_exp.iloc[:, 1].to_numpy() * (FARADAY * A * c_bulk * Dref / r_e) * 1e6
        ax_cv.plot(exp_potential, exp_current, "--", lw=2.5, color=colors[i])

    ax_cv.set_xlabel("E / V vs. SCE")
    ax_cv.set_ylabel(r"I / $\mu$A")
    ax_cv.legend()
    fig.savefig(output_dir / "PINN_paper_style.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved prediction plots in: {output_dir}")


def make_demo_data(output_dir: Path) -> list[str]:
    """Create toy dimensionless CSVs so the training script can be smoke-tested.

    These files are not physically rigorous; they only verify that the PyTorch
    training loop, autograd derivatives, interpolation, and plotting run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sigmas = [8.7623e2, 1.7525e3, 4.3811e3]
    theta_i = 1.8459e1
    theta_v = -3.0220e1
    dA = 4.63e-1
    files = []
    for sigma in sigmas:
        full_t = 2.0 * abs(theta_v - theta_i) / sigma
        t = np.linspace(0, full_t, 1000)
        theta = np.where(t < full_t / 2.0, theta_i - sigma * t, theta_v + sigma * (t - full_t / 2.0))
        # A smooth CV-like toy signal, scaled with sqrt(sigma).
        forward = -0.02 * np.sqrt(sigma) * np.exp(-0.5 * ((theta + 8.0) / 5.0) ** 2)
        reverse = 0.006 * np.sqrt(sigma) * np.exp(-0.5 * ((theta - 2.0) / 6.0) ** 2)
        flux = np.where(t < full_t / 2.0, forward, reverse)
        df = pd.DataFrame({"Potential": theta, "Flux": flux})
        name = output_dir / (
            f"Exp Dimensionless sigma={sigma:.4E} "
            f"theta_i={theta_i:.4E} theta_v={theta_v:.4E} dA={dA:.2E}.csv"
        )
        df.to_csv(name, index=False)
        files.append(str(name))
    print("Created demo data:")
    for f in files:
        print(f"  {f}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-files",
        nargs="+",
        default=[
            "ExpData/Exp Dimensionless sigma=8.7623E+02 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv",
            "ExpData/Exp Dimensionless sigma=1.7525E+03 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv",
            "ExpData/Exp Dimensionless sigma=4.3811E+03 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv",
        ],
    )
    parser.add_argument("--scan-rates", nargs="+", type=float, default=[0.01, 0.02, 0.05])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num-train-samples", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--portion-analyzed", type=float, default=0.75)
    parser.add_argument("--lambda-x", type=float, default=3.0)
    parser.add_argument("--outer-boundary", choices=["SI", "TL"], default="SI")
    parser.add_argument("--layers", nargs="+", type=int, default=[64, 32, 32, 32, 64])
    parser.add_argument("--output-dir", default="Epochs_pytorch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint used with --eval-only")
    parser.add_argument("--num-test-samples", type=int, default=400)
    parser.add_argument("--make-demo-data", action="store_true")
    parser.add_argument("--demo-data-dir", default="ExpDataDemo")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--torch-num-threads", type=int, default=1, help="Use 1 on CPU to avoid very slow higher-order autograd from thread overhead. Use 0 to leave PyTorch default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if args.make_demo_data:
        args.exp_files = make_demo_data(Path(args.demo_data_dir))
        args.scan_rates = [0.01, 0.02, 0.05]

    missing = [p for p in args.exp_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing experimental dimensionless CSV files:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun convert_to_dimensionless.py first, provide --exp-files, or use --make-demo-data for a smoke test."
        )

    parsed = [exp_parameters(p) for p in args.exp_files]
    sigmas = [p[0] for p in parsed]
    theta_i = parsed[0][1]
    theta_v = parsed[0][2]
    dA_file = parsed[0][3]

    print(f"Using device: {args.device}")
    print(f"Parsed sigmas: {sigmas}")
    print(f"theta_i={theta_i:.5g}, theta_v={theta_v:.5g}, dA_file={dA_file:.5g}")

    sampler = ECINNSampler(
        exp_file_names=args.exp_files,
        sigmas=sigmas,
        theta_i=theta_i,
        theta_v=theta_v,
        portion_analyzed=args.portion_analyzed,
        lambda_x=args.lambda_x,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = ECINN(num_experiments=len(args.exp_files), layers=args.layers).to(args.device)
    print(model)

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        state = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(state["model"] if "model" in state else state)
    else:
        steps_per_epoch = args.steps_per_epoch
        if steps_per_epoch is None:
            steps_per_epoch = max(1, args.num_train_samples // args.batch_size)
        train(
            model=model,
            sampler=sampler,
            output_dir=output_dir,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            lr=args.lr,
            outer_boundary=args.outer_boundary,
            resume=args.resume,
            save_every=args.save_every,
        )

    plot_predictions(
        model=model,
        sampler=sampler,
        output_dir=output_dir,
        scan_rates=args.scan_rates,
        num_test_samples=args.num_test_samples,
    )

    K0, alpha, dA = model.parameter_values()
    lambda_k0 = K0 * Dref / r_e
    lambda_k0_eff = lambda_k0 * math.exp(alpha * ERef * FARADAY / (R * T))
    print("\nFinal inferred values")
    print(f"  lambda_K0      = {K0:.6e}  [dimensionless]")
    print(f"  lambda_alpha   = {alpha:.6f}")
    print(f"  lambda_dA      = {dA:.6f}  [dimensionless]")
    print(f"  D              = {dA * Dref:.6e} m^2/s")
    print(f"  k0             = {lambda_k0:.6e} m/s")
    print(f"  k0_eff         = {lambda_k0_eff:.6e} m/s")


if __name__ == "__main__":
    main()
