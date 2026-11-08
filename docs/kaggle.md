# Running on Kaggle

The competition is a code competition: the submission notebook runs with internet disabled and sees the hidden test images only at scoring time. The same three commands as locally work there; only the inputs have to be attached in advance.

## One-off preparation (internet on)

1. **Code**: upload this repository as a private Kaggle Dataset (or attach it from GitHub).
2. **Backbone weights**: in a notebook with internet enabled, download the backbone once and save it as a **private** Kaggle Dataset.
   - DINOv3 (timm): save the checkpoint file (e.g. `model.safetensors` from the timm hub repo). Keep it private: the DINOv3 License does not allow re-publishing the weights.
   - SigLIP (transformers): `SiglipVisionModel.from_pretrained(...).save_pretrained(<dir>)` and upload `<dir>`.
3. **Python packages**: the Kaggle image already has torch, timm, transformers, scikit-learn and LightGBM. If a newer timm is needed for DINOv3, add its wheel to a dataset and install it with `pip install --no-index --find-links <wheel dir> timm`.

## Submission notebook (internet off)

```bash
cd <repo>
export PYTHONPATH=$PWD/src            # no install step needed
export BIOMASS_DATA_DIR=<competition input directory>
python scripts/extract_embeddings.py --set embed.weights=<checkpoint file or dir> \
       --set emb_dir=<writable dir>/emb
python scripts/run_cv.py --set emb_dir=<writable dir>/emb --set runs_dir=<writable dir>/runs
python scripts/make_submission.py --runs <writable dir>/runs/<run name> --out <writable dir>/submission.csv
```

- Train embeddings are recomputed in the notebook (357 images), so nothing derived from the data has to be uploaded.
- `make_submission.py` checks row count, duplicates, NaN, negative values and the additive identities before writing the file.
- Keep paths in notebook variables rather than in the repository; nothing here assumes a particular mount point.
