"""Convert Fe(III) CV Excel sheets to dimensionless ECINN CSV files.

Usage:
    python -m ecinn_pytorch.convert_to_dimensionless \
        --excel "5 mM Fe(3) 1 M H2SO4 on GCE.xlsx"

The output filenames match the TensorFlow example:
    ExpData/Exp Dimensionless sigma=... theta_i=... theta_v=... dA=....csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .helper import F, R, T

# Constants from the TensorFlow ConvertToDimensionless.py example.
r_e = 1.50e-3  # m
c_bulk = 4.85  # mol/m^3 = 4.85 mM
E_i = 1.0  # V vs SCE
E_v = -0.25  # V vs SCE
A = np.pi * r_e**2
DA = 4.63e-10
Dref = 1e-9
ERef = 0.526  # V, formal potential of Fe2+/Fe3+ couple used by original example

DEFAULT_SHEET_NAMES = ["10 mV s-1", "20 mV s-1", "50 mV s-1", "100 mV s-1", "200 mV s-1"][::-1]
DEFAULT_SCAN_RATES = [0.01, 0.02, 0.05, 0.1, 0.2][::-1]


def find_midpoint_potential(df: pd.DataFrame) -> float:
    df_forward = df.iloc[: int(len(df) / 2)]
    df_reverse = df.iloc[int(len(df) / 2) :].reset_index(drop=True)
    forward_scan_peak_potential = df_forward.iloc[df_forward.iloc[:, 1].idxmin(), 0]
    reverse_scan_peak_potential = df_reverse.iloc[df_reverse.iloc[:, 1].idxmax(), 0]
    return float((forward_scan_peak_potential + reverse_scan_peak_potential) / 2.0)


def convert_excel_to_dimensionless(
    excel_file: str | Path,
    output_dir: str | Path = "ExpData",
    sheet_names: list[str] | None = None,
    scan_rates: list[float] | None = None,
) -> list[Path]:
    excel_file = Path(excel_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sheet_names is None:
        sheet_names = DEFAULT_SHEET_NAMES
    if scan_rates is None:
        scan_rates = DEFAULT_SCAN_RATES
    if len(sheet_names) != len(scan_rates):
        raise ValueError("sheet_names and scan_rates must have the same length")

    written: list[Path] = []
    for nu, sheet_name in zip(scan_rates, sheet_names):
        df_exp = pd.read_excel(excel_file, sheet_name=sheet_name)

        sigma = (r_e**2 / Dref) * (F / (R * T)) * nu
        theta_i = (E_i - ERef) * (F / (R * T))
        theta_v = (E_v - ERef) * (F / (R * T))
        d_a = DA / Dref

        df_exp = df_exp.copy()
        df_exp.iloc[:, 0] = (df_exp.iloc[:, 0] - ERef) * (F / (R * T))
        df_exp.iloc[:, 1] = df_exp.iloc[:, 1] / (F * A * c_bulk * Dref / r_e)
        df_exp = df_exp.rename({"Potential(V)": "Potential", "Current(I)": "Flux"}, axis=1)

        out = output_dir / (
            f"Exp Dimensionless sigma={sigma:.4E} "
            f"theta_i={theta_i:.4E} theta_v={theta_v:.4E} dA={d_a:.2E}.csv"
        )
        df_exp.to_csv(out, index=False)
        written.append(out)
        print(f"Wrote {out}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", default="5 mM Fe(3) 1 M H2SO4 on GCE.xlsx")
    parser.add_argument("--output-dir", default="ExpData")
    args = parser.parse_args()
    convert_excel_to_dimensionless(args.excel, args.output_dir)


if __name__ == "__main__":
    main()
