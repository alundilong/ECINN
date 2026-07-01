# PyTorch ECINN for irreversible Fe(III) reduction

This is a PyTorch rewrite of the TensorFlow ECINN example you shared. It keeps the same physical structure:

- one neural network per scan rate, each approximating `c(t, x)`;
- shared trainable parameters `lambda_K0`, `lambda_alpha`, and `lambda_dA`;
- diffusion PDE residual, initial condition, electrode Butler--Volmer residual, experimental flux matching, and outer-boundary residual;
- gradual ramping of the BV boundary loss to avoid early instability from the exponential BV term.

## Install

```bash
pip install -r requirements.txt
```

## Convert the Excel file to dimensionless CSVs

Place your Excel file in the project root and run:

```bash
python -m ecinn_pytorch.convert_to_dimensionless --excel "5 mM Fe(3) 1 M H2SO4 on GCE.xlsx"
```

This creates files under `ExpData/` with names like:

```text
Exp Dimensionless sigma=8.7623E+02 theta_i=1.8459E+01 theta_v=-3.0220E+01 dA=4.63E-01.csv
```

## Train using the three scan rates from the TensorFlow example

```bash
python -m ecinn_pytorch.train_ecinn --epochs 300
```

For a quick test before a full run:

```bash
python -m ecinn_pytorch.train_ecinn --epochs 2 --steps-per-epoch 10 --num-test-samples 120
```

## Smoke test without real data

This generates nonphysical toy CSVs and verifies that the training loop runs:

```bash
python -m ecinn_pytorch.train_ecinn --make-demo-data --epochs 2 --steps-per-epoch 5 --num-test-samples 80
```

## Outputs

By default, results are written to `Epochs_pytorch/`:

- `history.csv`: losses and inferred parameters by epoch;
- `weights/ecinn_final.pt`: final PyTorch checkpoint;
- `PINN_scan_*.csv`: reconstructed dimensionless CVs;
- `PINN_dimensionless_summary.png`: concentration fields and dimensionless CVs;
- `PINN_paper_style.png`: dimensional paper-style figure.

## Notes

The default `--outer-boundary SI` matches the example's semi-infinite fixed-concentration boundary. The `TL` option is implemented as a no-flux condition `dc/dx=0`, which is physically appropriate for thin-layer behavior.
