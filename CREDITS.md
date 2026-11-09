# Credits and references

The code in this repository was newly written for it; no code, weights or data from other notebooks are included. The ideas it uses are credited below: first published methods, then ideas that also appear in the competition's public notebooks.

| idea | reference | use here |
|---|---|---|
| Dataset and task | Q. Liao, D. Wang, R. Haling et al., *Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture*, arXiv:2510.22916, 2025; Kaggle competition *CSIRO – Image2Biomass Prediction*, 2025 | task definition, target identities, metric |
| Self-supervised ViT features | O. Siméoni et al., *DINOv3*, arXiv:2508.10104, 2025 | main frozen backbone (via timm) |
| Image–text pretrained ViT | X. Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV 2023 | alternative frozen backbone (vision tower only) |
| Linear heads on frozen features | standard evaluation protocol for self-supervised models (e.g. M. Oquab et al., *DINOv2*, TMLR 2024) | Scaler + PCA + Ridge on embeddings |
| Grouped cross-validation for dependent samples | D. R. Roberts et al., *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*, Ecography 40(8), 2017 | folds grouped by sampling date |
| Ridge with efficient leave-one-out | scikit-learn `RidgeCV` (F. Pedregosa et al., JMLR 12, 2011) | alpha selection inside each fold under plain KFold (grouped schemes use grouped inner CV) |
| Gradient-boosted trees | G. Ke et al., *LightGBM*, NeurIPS 2017 | alternative head |
| Non-negative least squares | C. L. Lawson and R. J. Hanson, *Solving Least Squares Problems*, 1974; `scipy.optimize.nnls` | projection onto the additive constraints; stacking weights |
| Stacking with non-negative weights | L. Breiman, *Stacked Regressions*, Machine Learning 24, 1996 | cross-fit ensemble weights |
| Excess-green index | D. M. Woebbecke et al., *Color indices for weed identification under various soil, residue, and lighting conditions*, Trans. ASAE 38(1), 1995 | `color_stats` baseline features |
| Model zoo / loaders | R. Wightman, *PyTorch Image Models (timm)*; T. Wolf et al., *Transformers*, EMNLP 2020 (demos) | loading backbones |

## Ideas from the competition community

These ideas are shared in the [public notebooks of the competition](https://www.kaggle.com/competitions/csiro-biomass/code) and its discussions. They were re-implemented here from their description, not from code.

| idea | source | use here |
|---|---|---|
| Split the 2:1 image into left and right halves | public notebooks | same |
| Horizontally flipped views | public notebooks | modified: averaged into the embedding for train and test, not only at test time |
| Frozen backbone embeddings with dimensionality reduction and a light regressor | public notebooks | same idea; PCA + Ridge / LightGBM (no PLS) |
| Predict the three components and sum them | public notebooks | same (`sum`) |
| Predict Total, GDM and Green and derive the rest by differences | public notebooks | same (`diff`), with negative differences set to 0 and parents rebuilt |
| Project predictions onto the additive constraints | public notebooks | modified: metric-weighted, non-negative projection (`nnls`) |
| Group CV folds by sampling date | public notebooks and discussions | same (`group_date`) |
| Group CV folds by month | Kinosuke, *草コンペ振り返り* (competition retrospective), Speaker Deck, 2026, https://speakerdeck.com/chihironakayama/cao-konhezhen-rifan-ri | modified: month × State (`group_month_state`) |
| Image–text similarity features from SigLIP | public notebooks | not used (vision tower only) |
| Test-time image editing, clipping to fixed per-target maxima, leaderboard-tuned multipliers | public notebooks | not used |

Third-party license notes (data, DINOv3, SigLIP) are in the README.
