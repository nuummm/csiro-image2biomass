# csiro-image2biomass

Predicting five pasture-biomass quantities from top-view photos with a frozen DINOv3 backbone and light regression heads.
The repository focuses on evaluation design: date-grouped cross-validation, four ways to respect the additive relations between targets, and OOF-based ensembling.
Everything runs from three commands once the competition data is on disk; a synthetic dataset lets the whole pipeline run on a CPU without it.

[![CI](https://github.com/nuummm/csiro-image2biomass/actions/workflows/ci.yml/badge.svg)](https://github.com/nuummm/csiro-image2biomass/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Context: written after the competition, in which I placed 253rd of 3,805 teams (bronze medal, solo); see [Background](#background).

## TL;DR

- **Features**: frozen DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`) embeddings of the left and right halves of each image plus their mirror images, averaged per image. ViT-L/16 and SigLIP so400m are wired in but have not been run.
- **Heads**: Scaler + PCA fitted inside each fold, then Ridge (alpha by grouped inner CV on the training part; efficient leave-one-out only under plain KFold) or LightGBM (tree count by early stopping on a grouped hold-out, then refit on the whole training part).
- **CV**: random KFold vs. `GroupKFold` on `Sampling_Date` (and month × State), each repeated over several seeds, so the gap between the two is reported with a spread.
- **Outputs**: 5 independent models / sum of 3 components / differences of the top 3 / weighted NNLS projection onto the feasible set.
- **Results** (357 training images, 5 folds × 3 seeds, OOF weighted R², mean ± std over seeds): the base setup (Ridge on `log1p` targets, NNLS projection) scores **0.736 ± 0.011** under date-grouped CV and 0.812 ± 0.008 under random KFold, so a random split reports about 0.08 more than the date-grouped estimate. The NNLS projection beats predicting the five targets independently (0.722 ± 0.010), and stacking does not clearly beat the single base model (0.737 ± 0.011). Details in [Results](#results) and [`reports/results.md`](reports/results.md).

## Task

[CSIRO – Image2Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass) (Kaggle). Each image shows a 70 cm × 30 cm pasture quadrat and is resampled to 2,000 × 1,000 px (dataset paper). For every image, predict dry biomass in grams for

| target | meaning | metric weight |
|---|---|---|
| `Dry_Green_g` | green vegetation excluding clover | 0.1 |
| `Dry_Dead_g` | dead material | 0.1 |
| `Dry_Clover_g` | clover | 0.1 |
| `GDM_g` | green dry matter = Green + Clover | 0.2 |
| `Dry_Total_g` | total = GDM + Dead | 0.5 |

Files are in long format: one row per `(image, target)` with `sample_id = <image_id>__<target_name>`. The training CSV also has `Sampling_Date`, `State`, `Species`, NDVI and plate height; the test CSV has only the image path, so these columns are used for grouping and EDA but never as model inputs. The dataset paper reports 1,162 images in total; the public training split has 357 of them and the rest form the hidden test set.

**Metric** (Evaluation page). A single R² over *all* rows together, each row weighted by its target, with the global weighted mean as the baseline:

$$
R^2_w = 1 - \frac{\sum_j w_j (y_j - \hat y_j)^2}{\sum_j w_j (y_j - \bar y_w)^2},
\qquad \bar y_w = \frac{\sum_j w_j y_j}{\sum_j w_j}.
$$

This is not the average of per-target R² values: because the baseline is shared, targets with large values (Total) dominate both the numerator and the denominator. The shared baseline also puts the differences between target means into the denominator, so a constant per-target prediction (each target's mean over the evaluated rows) already scores above 0 whenever the target means differ; the useful reference point is that constant, not 0. [`metric.py`](src/biomass/metric.py) implements this definition and a test checks it against the long-format formula.

## Pipeline

```mermaid
flowchart LR
    A[image 2000x1000] --> B[left / right halves<br/>+ horizontal flips]
    B --> C[frozen backbone<br/>CLS + mean patch token]
    C --> D[mean over views<br/>= image embedding]
    D --> E[fold-local<br/>Scaler + PCA]
    E --> F[Ridge / LightGBM<br/>raw or log1p target]
    F --> G[inverse transform<br/>to grams]
    G --> H[output formulation<br/>independent / sum / diff / NNLS]
    H --> I[OOF score<br/>+ test prediction]
    I --> J[ensemble<br/>mean / best / cross-fit NNLS]
```

## CV design

The training images come from a limited number of sampling events (date × site). Photos from the same event share light, season and pasture state, so a random split puts near-duplicates on both sides and is expected to look better than it should. [`folds.py`](src/biomass/folds.py) provides

- `kfold` – shuffled KFold (reference point),
- `group_date` – every `Sampling_Date` stays inside one fold,
- `group_month_state` – coarser groups (month × State), used only when there are enough groups.

Grouped folds are built by visiting groups in a seeded random order and assigning each to the currently smallest fold; the function asserts that no group spans two folds. Every scheme is repeated over the seeds in `configs/base.yaml`, and the report gives mean ± std over seeds as well as over individual folds. `scripts/inspect_data.py` prints the number of groups before any of this is trusted.

Leakage control: the scaler, PCA, Ridge's alpha search and LightGBM's early stopping only see the training part of each fold. Under a grouped scheme the alpha search uses grouped inner folds and the early-stopping hold-out keeps groups together too, because leave-one-out or a random hold-out would again put images of one sampling event on both sides. `tests/test_no_leak.py` checks that the fitted scaler mean equals the training-part mean in every fold, that the grouped alpha search is not fooled by group effects, and that the hold-out never splits a group.

## Output formulation

All five targets are determined by three non-negative components $c = (\text{Green}, \text{Dead}, \text{Clover})$ through $y = Ac$:

$$
A = \begin{pmatrix} 1&0&0\\ 0&1&0\\ 0&0&1\\ 1&0&1\\ 1&1&1 \end{pmatrix}.
$$

| formulation | fitted targets | how the five outputs are formed | identities hold? |
|---|---|---|---|
| `independent` | all 5 | clip at 0 | no |
| `sum` | Green, Dead, Clover | $\hat y = A\,\max(\hat c, 0)$ | yes |
| `diff` | Total, GDM, Green | Clover = GDM − Green, Dead = Total − GDM (negative differences set to 0, parents rebuilt) | yes |
| `nnls` | all 5 | $\hat c = \arg\min_{c \ge 0} \lVert W^{1/2}(Ac - \hat y)\rVert^2$, output $A\hat c$ | yes |

The trade-off: `sum` accumulates the component errors into Total (weight 0.5), `diff` pushes the errors of large quantities into the small ones, and `nnls` uses all five predictions. With $W$ equal to the metric weights, the NNLS step finds the feasible point that is closest in exactly the weighted squared error that enters the metric's residual term. Because the feasible set is convex, this projection is non-expansive in the same weighted norm: whenever the true values satisfy the identities, the projected prediction of an image is never farther from them than the raw prediction (a test checks this on random data). Formulations are applied on the gram scale, after `expm1` when the model was trained on `log1p` targets ([`constraints.py`](src/biomass/constraints.py); tests check the identities to 1e-6 and that a feasible input is left unchanged).

## Target space and ensembling

- **Target space**: `raw` or `log1p`. The targets are skewed, which favours `log1p` for fitting, but the metric is on the raw scale and dominated by large Total values, where `log1p` models tend to under-predict. Both are evaluated on OOF predictions rather than chosen in advance.
- **Ensembling** ([`ensemble.py`](src/biomass/ensemble.py)): simple mean, best single run, and *cross-fit* NNLS stacking. For the last two, the choice (or the weights) is made on the other folds' OOF predictions and applied to the held-out fold, so the reported ensemble score is not fitted on the rows it is evaluated on; the weights used on the test set are fitted on all rows. One non-negative weight per model is shared across the five targets; a non-negative combination of predictions that satisfy the identities still satisfies them, so no re-projection is needed. The fold-to-fold spread of the weights is reported too: if stacking does not clearly beat the mean, the mean is the safer choice.

## Ablations

`scripts/run_all.py` starts from `configs/base.yaml` and changes one setting at a time:

| axis | levels |
|---|---|
| CV scheme | kfold / group_date / group_month_state |
| backbone | dinov3_base / color_stats (hand-crafted colour baseline); dinov3_large and siglip_so400m are implemented but not run |
| head | ridge / lgbm |
| target space | raw / log1p |
| output formulation | independent / sum / diff / nnls |
| ensemble | mean / best / cross-fit NNLS |

## Results

All numbers below were produced by `scripts/run_all.py` on the competition's training data (357 images; the test labels are hidden, so everything is out-of-fold). Base configuration: `dinov3_base`, `group_date` CV with 5 folds and seeds 0, 1, 2, Ridge on `log1p` targets, NNLS projection. Each row changes one setting; the score is the competition metric on OOF predictions, mean ± std over the three seeds.

| axis | setting | OOF weighted R² |
|---|---|---|
| CV scheme | random KFold | 0.812 ± 0.008 |
| | **group by `Sampling_Date`** (base) | **0.736 ± 0.011** |
| | group by month × State | 0.717 ± 0.009 |
| features | **DINOv3 ViT-B/16** (base) | **0.736 ± 0.011** |
| | colour statistics (12 hand-crafted features) | 0.324 ± 0.023 |
| head | **Ridge** (base) | **0.736 ± 0.011** |
| | LightGBM | 0.620 ± 0.016 |
| target space | raw | 0.741 ± 0.010 |
| | **log1p** (base) | **0.736 ± 0.011** |
| output formulation | independent | 0.722 ± 0.010 |
| | sum of 3 components | 0.575 ± 0.009 |
| | differences of the top 3 | 0.689 ± 0.012 |
| | **NNLS projection** (base) | **0.736 ± 0.011** |
| ensemble of 4 runs | mean | 0.686 ± 0.003 |
| (cross-fit, see above) | best single run | 0.742 ± 0.015 |
| | NNLS stacking | 0.737 ± 0.011 |

![OOF weighted R² by CV scheme](reports/figures/cv_scheme.png)

What the numbers say, and what they do not:

- **CV design matters most.** Random KFold reports about 0.08 more than date-grouped CV with the same model; the gap is several times the seed-to-seed spread. Grouping more coarsely (month × State, 18 groups) lowers the score a little further.
- **Respecting the identities helps only with the right formulation.** The NNLS projection is better than independent predictions in each of the three seeds (by 0.012 to 0.016 on the same folds); building everything from three predicted components (`sum`) is clearly worse, because the errors of the components add up in Total.
- **raw vs. log1p** differ by less than one standard deviation, so neither is preferred on this evidence.
- **Ridge beats LightGBM** on these 128 PCA components with 357 images and default LightGBM settings; LightGBM was not tuned.
- **Ensembling** (members: the base run, the colour baseline, LightGBM and the raw-target run): the plain mean is pulled down by the weak colour baseline; cross-fit NNLS stacking ends up at the base model's level. With four members, one of them weak, the base model alone is the simpler choice.
- The ± values only reflect how the images are assigned to folds. With 28 sampling dates, a difference of about 0.01 between two rows should not be read as real unless it holds seed by seed on the same folds. These are CV scores on the training set, not leaderboard scores; this code has not been scored on the hidden test set.

Per-target R², the Total calibration plot and the stacking weights are in [`reports/results.md`](reports/results.md). Embedding extraction took about 2 minutes on an Apple M5 GPU (MPS, fp32) and all CV runs about 1 minute on its CPU.

## Reproduce

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.
On macOS, LightGBM also needs OpenMP: `brew install libomp` (without it the LightGBM tests fail to import the library).

```bash
uv sync --extra embed            # torch, timm, transformers for feature extraction
```

Download the data from the competition page after accepting its rules and point the code at it (the directory with `train.csv`, `test.csv`, `train/`, `test/`):

```bash
export BIOMASS_DATA_DIR=/path/to/csiro-biomass
uv run python scripts/inspect_data.py                  # group counts, identity residuals
uv run python scripts/extract_embeddings.py            # 1. features -> emb/<backbone>/
uv run python scripts/run_cv.py                        # 2. CV -> runs/<name>/
uv run python scripts/make_submission.py --runs runs/dinov3_base-group_date-ridge-log1p-nnls    # 3.
uv run python scripts/run_all.py                       # all ablations -> reports/results.md
```

The device is picked automatically (CUDA, then Apple MPS, then CPU) and can be forced with `--device` or `BIOMASS_DEVICE`. Settings are changed with `--set`, e.g. `--set backbone=siglip_so400m --set model.head=lgbm`. Running on Kaggle (internet disabled) is described in [`docs/kaggle.md`](docs/kaggle.md).

**Without the competition data**:

```bash
uv sync
uv run python scripts/make_synthetic_data.py --out data/synthetic
export BIOMASS_DATA_DIR=data/synthetic
uv run python scripts/extract_embeddings.py --set backbone=color_stats \
    --set emb_dir=emb/synthetic
uv run python scripts/run_all.py --set backbone=color_stats \
    --set emb_dir=emb/synthetic --set runs_dir=runs/synthetic \
    --set model.pca_components=8 --report-dir runs/synthetic/report
```

The synthetic images are noise whose colours follow made-up targets; the scores they produce say nothing about the real task. Separate `emb_dir` / `runs_dir` keep them apart from real runs; `run_all.py` also recomputes any run whose saved config or training images differ from the current ones.

**Tests**: `uv run pytest` (CPU only, synthetic inputs). The transformer tests build DINOv3 and SigLIP with random weights (`pretrained=False`), so nothing is downloaded; they are skipped when the `[embed]` extra is not installed, as in CI.

**Reproducibility boundary**: seeds fix the folds, PCA and LightGBM, and `uv.lock` pins the Python packages. Embeddings can still differ slightly across hardware and fp16 (CUDA) vs fp32 (MPS / CPU), and the backbone weights are fetched from their publishers rather than shipped here.

## Background

My bronze-medal submission to Kaggle CSIRO Image2Biomass (253rd of 3,805 teams, private LB 0.61366, solo) was built by combining public notebooks of the competition. This repository does not use that submission, or the code or weights of those notebooks: it is an independent implementation written anew after the competition, and every number it reports comes from its own code, so its results are not comparable to that submission. Several modelling ideas used here also appear in the competition's public notebooks; [CREDITS.md](CREDITS.md) lists them.

## Limitations and future work

- Features are frozen; fine-tuning the backbone was out of scope.
- No use of text prompts / image–text similarity from SigLIP; only the vision tower is used.
- No PLS, Gaussian-mixture features, CatBoost, or hurdle models for the many zero-clover images.
- No target-specific loss weighting, pseudo-labelling, or test-time image editing.
- Hyperparameters (PCA size, alpha grid, LightGBM settings) are reasonable defaults, not tuned.
- Only one pretrained backbone (DINOv3 ViT-B/16) has been evaluated; comparing it with ViT-L/16 and SigLIP so400m is the next step.
- The pipeline has only been evaluated by cross-validation on the 357 training images; it has not been scored on the hidden test set.

## Project layout

```
configs/base.yaml          default experiment (every key can be overridden with --set)
src/biomass/
  data.py                  long <-> wide, sample_id, paths from config / $BIOMASS_DATA_DIR
  folds.py                 KFold / date-grouped / month x State folds, repeated over seeds
  metric.py                global weighted R^2 (competition metric)
  constraints.py           independent / sum / diff / NNLS output formulations
  embed.py                 views, frozen DINOv3 (timm) / SigLIP (transformers), device choice
  cv.py                    fold-local preprocessing, Ridge / LightGBM, OOF + test predictions
  ensemble.py              mean / best / cross-fit NNLS
  submit.py                long-format submission + validation
  pipeline.py, config.py   glue for the scripts
  synthetic.py             fake dataset for tests and dry runs
scripts/                   inspect_data, extract_embeddings, run_cv, make_submission, run_all
tests/                     metric, constraints, folds, no_leak, ensemble, submit, data, config, embed, smoke
reports/                  results.md and figures/, written by scripts/run_all.py
```

## License and third-party notices

Code: MIT, see [LICENSE](LICENSE). References for the methods are in [CREDITS.md](CREDITS.md).

- **Data**: the competition data (CC BY-SA 4.0, also subject to the competition rules) is not included. Dataset paper: Liao et al., *Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture*, arXiv:2510.22916 (2025).
- **DINOv3** weights are released by Meta under the DINOv3 License and are not included or redistributed here; neither are embeddings computed from them. Work that uses them should acknowledge DINOv3 as the license requires.
- **SigLIP** (`google/siglip-so400m-patch14-384`) is released under Apache-2.0; weights are not included.
- Libraries are used under their own licenses (timm and transformers: Apache-2.0; scikit-learn: BSD-3-Clause; LightGBM: MIT; SciPy, NumPy, pandas: BSD-3-Clause).

## 日本語要約

牧草の真上写真から 5 種類のバイオマス量を推定する Kaggle コンペ（CSIRO Image2Biomass）を題材に、凍結した DINOv3（ViT-B/16）の埋め込みと軽い回帰ヘッドで学習パイプラインを実装したものです。日付でグループ化した CV と通常の KFold の差、ターゲット間の加法関係の 4 通りの扱い方、OOF に基づくアンサンブルを比較できるようにしています。コンペでは銅メダル（253 位 / 3,805 チーム、private 0.61366、個人参加）で、その提出は公開ノートブックを組み合わせたものでした。この repo はその提出や公開ノートブックのコード・重みを使わずにコンペ後に新しく書いた独立実装で、数値はすべてこの repo のコードによるものです（コンペ提出とは比較できません）。公開ノートブックにも見られるアイデアは CREDITS.md に記載しています。学習データ 357 枚での OOF 評価（5 fold × 3 seed、平均 ± 標準偏差）では、基本構成（Ridge、log1p、NNLS 射影）が日付グループ CV で 0.736 ± 0.011、ランダム KFold で 0.812 ± 0.008 となり、ランダム分割は約 0.08 高く見積もられました。NNLS 射影は 5 ターゲット独立予測（0.722 ± 0.010）を上回り、スタッキング（0.737 ± 0.011）は単体モデルと差がありませんでした。隠しテストセットでの評価はしていません。
