# Credits and references

All code in this repository was written for it. The techniques it combines are well known; the table lists where they come from.

| idea | reference | use here |
|---|---|---|
| Dataset and task | Q. Liao et al., *Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture*, arXiv:2510.22916, 2025; Kaggle competition *CSIRO – Image2Biomass Prediction*, 2025 | task definition, target identities, metric |
| Self-supervised ViT features | O. Siméoni et al., *DINOv3*, arXiv:2508.10104, 2025 | main frozen backbone (via timm) |
| Image–text pretrained ViT | X. Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV 2023 | alternative frozen backbone (vision tower only) |
| Linear heads on frozen features | standard evaluation protocol for self-supervised models (e.g. M. Oquab et al., *DINOv2*, TMLR 2024) | Scaler + PCA + Ridge on embeddings |
| Grouped cross-validation for dependent samples | D. R. Roberts et al., *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*, Ecography 40(8), 2017 | folds grouped by sampling date |
| Ridge with efficient leave-one-out | scikit-learn `RidgeCV` (F. Pedregosa et al., JMLR 12, 2011) | alpha selection inside each fold |
| Gradient-boosted trees | G. Ke et al., *LightGBM*, NeurIPS 2017 | alternative head |
| Non-negative least squares | C. L. Lawson and R. J. Hanson, *Solving Least Squares Problems*, 1974; `scipy.optimize.nnls` | projection onto the additive constraints; stacking weights |
| Stacking with non-negative weights | L. Breiman, *Stacked Regressions*, Machine Learning 24, 1996 | cross-fit ensemble weights |
| Excess-green index | D. M. Woebbecke et al., *Color indices for weed identification under various soil, residue, and lighting conditions*, Trans. ASAE 38(1), 1995 | `color_stats` baseline features |
| Model zoo / loaders | R. Wightman, *PyTorch Image Models (timm)*; T. Wolf et al., *Transformers*, EMNLP 2020 (demos) | loading backbones |

Third-party license notes (data, DINOv3, SigLIP) are in the README.
