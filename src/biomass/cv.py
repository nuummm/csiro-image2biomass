"""Cross-validated training of light heads on frozen embeddings.

Inside each fold, everything that learns from data (scaler, PCA, alpha search,
early stopping) only sees the training part of that fold. Predictions are
moved back to the raw scale, then the output formulation is applied, so the
OOF matrix is directly comparable with the competition metric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from biomass.constraints import FORMULATIONS, compose, fitted_columns
from biomass.metric import r2_per_target, weighted_r2

HEADS = ("ridge", "lgbm")
TARGET_SPACES = ("raw", "log1p")


@dataclass
class CVConfig:
    head: str = "ridge"
    target_space: str = "log1p"
    formulation: str = "nnls"
    nnls_weights: str = "metric"
    pca_components: int | None = 128
    ridge_alphas: list[float] = field(default_factory=lambda: np.logspace(-2, 4, 13).tolist())
    lgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 10,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        }
    )
    lgbm_early_stopping: int = 100
    lgbm_holdout: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}")
        if self.target_space not in TARGET_SPACES:
            raise ValueError(f"target_space must be one of {TARGET_SPACES}")
        if self.formulation not in FORMULATIONS:
            raise ValueError(f"formulation must be one of {FORMULATIONS}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FoldModel:
    preprocessor: Pipeline
    heads: list
    train_idx: np.ndarray


# --------------------------------------------------------------------------- pieces


def to_model_space(y: np.ndarray, space: str) -> np.ndarray:
    return np.log1p(np.clip(y, 0.0, None)) if space == "log1p" else y


def to_raw_space(y: np.ndarray, space: str) -> np.ndarray:
    return np.expm1(y) if space == "log1p" else y


def predict_raw(fitted_model_space: np.ndarray, cfg: CVConfig) -> np.ndarray:
    """Inverse target transform first, then the output formulation (raw scale)."""
    raw = np.clip(to_raw_space(fitted_model_space, cfg.target_space), 0.0, None)
    return compose(cfg.formulation, raw, nnls_weights=cfg.nnls_weights)


def make_preprocessor(n_train: int, dim: int, cfg: CVConfig) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if cfg.pca_components:
        n_comp = min(cfg.pca_components, n_train - 1, dim)
        steps.append(("pca", PCA(n_components=n_comp, random_state=cfg.seed)))
    return Pipeline(steps)


def _fit_heads(z, y_fit, groups, cfg: CVConfig) -> list:
    if cfg.head == "ridge":
        model = RidgeCV(alphas=cfg.ridge_alphas, alpha_per_target=True)
        model.fit(z, y_fit)
        return [model]

    import lightgbm as lgb

    if groups is not None:
        splitter = GroupShuffleSplit(1, test_size=cfg.lgbm_holdout, random_state=cfg.seed)
        tr, es = next(splitter.split(z, groups=groups))
    else:
        splitter = ShuffleSplit(1, test_size=cfg.lgbm_holdout, random_state=cfg.seed)
        tr, es = next(splitter.split(z))
    heads = []
    for j in range(y_fit.shape[1]):
        model = lgb.LGBMRegressor(**cfg.lgbm_params, random_state=cfg.seed, verbose=-1)
        model.fit(
            z[tr],
            y_fit[tr, j],
            eval_X=(z[es],),
            eval_y=(y_fit[es, j],),
            callbacks=[lgb.early_stopping(cfg.lgbm_early_stopping, verbose=False)],
        )
        heads.append(model)
    return heads


def _predict_heads(heads: list, z: np.ndarray, cfg: CVConfig) -> np.ndarray:
    if cfg.head == "ridge":
        return heads[0].predict(z)
    return np.column_stack([h.predict(z, num_iteration=h.best_iteration_) for h in heads])


def fit_fold(X, Y, train_idx, cfg: CVConfig, groups=None) -> FoldModel:
    pre = make_preprocessor(len(train_idx), X.shape[1], cfg)
    z = pre.fit_transform(X[train_idx])
    y_fit = to_model_space(Y[train_idx][:, fitted_columns(cfg.formulation)], cfg.target_space)
    g = None if groups is None else np.asarray(groups)[train_idx]
    return FoldModel(pre, _fit_heads(z, y_fit, g, cfg), np.asarray(train_idx))


def predict_fold(model: FoldModel, X, cfg: CVConfig) -> np.ndarray:
    return predict_raw(_predict_heads(model.heads, model.preprocessor.transform(X), cfg), cfg)


# --------------------------------------------------------------------------- loops


@dataclass
class CVResult:
    oof: np.ndarray
    test: np.ndarray | None
    fold_scores: list[float]
    score: float
    models: list[FoldModel] | None = None


def run_cv(
    X: np.ndarray,
    Y: np.ndarray,
    folds: np.ndarray,
    cfg: CVConfig,
    X_test: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    keep_models: bool = False,
) -> CVResult:
    """One pass over ``folds``: raw-scale OOF and fold-averaged test predictions."""
    folds = np.asarray(folds)
    oof = np.full(Y.shape, np.nan)
    test_preds, fold_scores, models = [], [], []
    for k in np.unique(folds):
        val = np.where(folds == k)[0]
        train = np.where(folds != k)[0]
        model = fit_fold(X, Y, train, cfg, groups)
        oof[val] = predict_fold(model, X[val], cfg)
        fold_scores.append(weighted_r2(Y[val], oof[val]))
        if X_test is not None:
            test_preds.append(predict_fold(model, X_test, cfg))
        if keep_models:
            models.append(model)
    if np.isnan(oof).any():
        raise RuntimeError("some rows never appeared in a validation fold")
    test = np.mean(test_preds, axis=0) if test_preds else None
    return CVResult(oof, test, fold_scores, weighted_r2(Y, oof), models if keep_models else None)


def run_repeated(
    X: np.ndarray,
    Y: np.ndarray,
    fold_table,
    cfg: CVConfig,
    X_test: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> dict:
    """Run :func:`run_cv` once per fold column (one per seed) and summarise."""
    oofs, tests, scores, fold_scores = {}, [], {}, []
    for col in fold_table.columns:
        res = run_cv(X, Y, fold_table[col].to_numpy(), cfg, X_test, groups)
        oofs[col] = res.oof
        scores[col] = res.score
        fold_scores.extend(res.fold_scores)
        if res.test is not None:
            tests.append(res.test)
    per_target = [r2_per_target(Y, o) for o in oofs.values()]
    summary = {
        "oof_score_mean": float(np.mean(list(scores.values()))),
        "oof_score_std": float(np.std(list(scores.values()))),
        "fold_score_mean": float(np.mean(fold_scores)),
        "fold_score_std": float(np.std(fold_scores)),
        "oof_score_by_seed": scores,
        "r2_per_target": {t: float(np.mean([p[t] for p in per_target])) for t in per_target[0]},
    }
    return {
        "summary": summary,
        "oof": oofs,
        "test": np.mean(tests, axis=0) if tests else None,
    }
