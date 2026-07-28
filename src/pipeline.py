import logging
from src.data import load_config, load_raw, clean, build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def run_pipeline():
    cfg = load_config()
    log = logging.getLogger("pipeline")

    log.info("Ingesting raw data…")
    raw = load_raw(cfg)

    log.info("Cleaning data…")
    cleaned = clean(raw, cfg)

    log.info("Engineering features…")
    features = build_features(cleaned, cfg)

    out = cfg["data"]["processed_path"]
    # ensure parent dir exists
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    features.to_parquet(out, index=False)
    log.info(f"Saved {len(features)} customer features → {out}")
    log.info(f"Churn distribution:\n{features['churn'].value_counts(normalize=True)}")


if __name__ == "__main__":
    run_pipeline()
