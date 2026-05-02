# Evaluation Results

## Mean ± Std Metrics

| Dataset | Model | Tier | RMSE | MAE | R^2 | Train(s) | PeakMem(MB) | Refresh | Cond(kappa) |
|---------|-------|------|------|-----|-----|----------|-------------|---------|-------------|
| WineQuality | AwareKernel | Small | 0.834624 ± 0.000000 | 0.638284 ± 0.000000 | 0.240251 ± 0.000000 | 6.2105 ± 0.0000 | 170.55 ± 0.00 | 1.0 ± 0.0 | 1.49e+04 ± 0.00e+00 |
| WineQuality | Ridge | Small | 0.822297 ± 0.000000 | 0.641703 ± 0.000000 | 0.262528 ± 0.000000 | 0.0002 ± 0.0000 | 0.01 ± 0.00 | 0.0 ± 0.0 | nan ± nan |
| WineQuality | Nystrom | Small | 0.884019 ± 0.000000 | 0.688999 ± 0.000000 | 0.147662 ± 0.000000 | 0.0426 ± 0.0000 | 13.59 ± 0.00 | 0.0 ± 0.0 | nan ± nan |
| WineQuality | RFF | Small | 0.791505 ± 0.000000 | 0.622285 ± 0.000000 | 0.316725 ± 0.000000 | 0.1180 ± 0.0000 | 35.63 ± 0.00 | 0.0 ± 0.0 | nan ± nan |

## Stability-over-Refits

| Dataset | Model | Tier | Var(RMSE) | Var(Time) | Var(Cond) |
|---------|-------|------|-----------|-----------|-----------|
| WineQuality | AwareKernel | Small | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| WineQuality | Ridge | Small | 0.000000e+00 | 0.000000e+00 | nan |
| WineQuality | Nystrom | Small | 0.000000e+00 | 0.000000e+00 | nan |
| WineQuality | RFF | Small | 0.000000e+00 | 0.000000e+00 | nan |