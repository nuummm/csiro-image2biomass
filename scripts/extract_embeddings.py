"""Step 1: cache frozen-backbone features for train and test images.

uv run python scripts/extract_embeddings.py --set backbone=dinov3_large
"""

from __future__ import annotations

import argparse
import time

from biomass.config import load_config
from biomass.data import load_test, load_train
from biomass.embed import BACKBONES, build_encoder, extract
from biomass.pipeline import save_embeddings


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument(
        "--set", action="append", default=[], help="override, e.g. backbone=siglip_so400m"
    )
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    backbone = cfg["backbone"]
    spec = BACKBONES[backbone]
    encoder = build_encoder(
        backbone, pretrained=True, device=args.device, weights=cfg["embed"].get("weights")
    )

    for split in args.splits:
        frame = (
            load_train(cfg.get("data_dir")) if split == "train" else load_test(cfg.get("data_dir"))
        )
        start = time.perf_counter()
        print(f"[{backbone}] {split}: {len(frame)} images", flush=True)
        X = extract(
            frame["image_path"].tolist(),
            encoder,
            spec.image_size,
            batch_size=cfg["embed"]["batch_size"],
            split_halves=cfg["embed"]["split_halves"],
            flip=cfg["embed"]["flip"],
            progress=True,
        )
        meta = {
            "backbone": backbone,
            "model_name": spec.model_name,
            "image_size": spec.image_size,
            **{k: v for k, v in cfg["embed"].items() if k != "weights"},
        }
        path = save_embeddings(cfg["emb_dir"], backbone, split, frame["image_id"].tolist(), X, meta)
        print(f"  -> {path} {X.shape} in {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
