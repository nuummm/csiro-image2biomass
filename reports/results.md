# Results

Not generated yet. This file is overwritten by `scripts/run_all.py` once the competition data is available (see the README, *Reproduce*). The tables below show the layout; no value here comes from a run.

Scores: competition metric (global weighted R²) on out-of-fold predictions, mean ± std over CV seeds.

## CV scheme

| cv.scheme | OOF R² | fold R² | R² Total |
|---|---|---|---|
| kfold | | | |
| group_date | | | |
| group_month_state | | | |

## Backbone

| backbone | OOF R² | fold R² | R² Total |
|---|---|---|---|
| dinov3_large | | | |
| siglip_so400m | | | |
| color_stats | | | |

## Head

| model.head | OOF R² | fold R² | R² Total |
|---|---|---|---|
| ridge | | | |
| lgbm | | | |

## Target space

| model.target_space | OOF R² | fold R² | R² Total |
|---|---|---|---|
| raw | | | |
| log1p | | | |

## Output formulation

| model.formulation | OOF R² | fold R² | R² Total |
|---|---|---|---|
| independent | | | |
| sum | | | |
| diff | | | |
| nnls | | | |

## Per-target R² (base config, first seed)

| Dry_Green_g | Dry_Dead_g | Dry_Clover_g | GDM_g | Dry_Total_g |
|---|---|---|---|---|
| | | | | |

## Ensembling

| method | OOF R² | weights |
|---|---|---|
| mean | | |
| best | | |
| crossfit_nnls | | |
