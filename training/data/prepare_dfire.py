"""
Downloads the D-Fire dataset and remaps its class ids to match this
project's labelmap (fire=1, smoke=2 — person=0 is reserved for the COCO
subset).

IMPORTANT — why this version is different from the original:
The GitHub repo referenced by the old `dfire.repo_url` config key
(gaia-solutions-on-demand/DFireDataset) does NOT contain the actual images
and labels. It only ships a README, a LICENSE, an example figure, and a
utils.py helper. The real dataset is distributed separately via a OneDrive
link in that README. `git clone`-ing the repo used to succeed silently and
produce zero image/label pairs, which meant training silently ran on
person-only data while still reporting "fire" and "smoke" in the class list.

This version downloads from the Kaggle mirror instead (much easier to
script than OneDrive), and — critically — raises an error if it ends up
with zero labeled images for any expected class, instead of silently
continuing with an incomplete dataset.

Setup (one-time):
    pip install kagglehub --break-system-packages
    Get a Kaggle API token at https://www.kaggle.com/settings -> "Create New Token"
    Set credentials via one of:
      - env vars: KAGGLE_USERNAME=... KAGGLE_KEY=...
      - or a kaggle.json file at ~/.kaggle/kaggle.json (downloaded from the settings page)

Config (training/configs/train_config.yaml) — replace the old `dfire.repo_url`
key with:
    dfire:
      kaggle_dataset: "sayedgamal99/smoke-fire-detection-yolo"
      source_class_map:
        0: smoke
        1: fire

Run:
    python training/data/prepare_dfire.py
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from common import load_config, resolve


def download_dataset(kaggle_dataset: str) -> Path:
    """Downloads the dataset via kagglehub and returns the local path it was cached to."""
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "kagglehub is not installed. Run:\n"
            "    pip install kagglehub --break-system-packages\n"
            "and make sure your Kaggle API credentials are set "
            "(KAGGLE_USERNAME/KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json)."
        )

    print(f"Downloading Kaggle dataset '{kaggle_dataset}' (this is cached locally after the first run)...")
    path = kagglehub.dataset_download(kaggle_dataset)
    return Path(path)


def find_image_label_dirs(root: Path):
    """
    Yields every (images_dir, labels_dir) pair found anywhere under root.
    Handles datasets that nest train/val/test splits at arbitrary depth.
    """
    for images_dir in root.rglob("images"):
        labels_dir = images_dir.parent / "labels"
        if labels_dir.exists():
            yield images_dir, labels_dir


def remap_and_copy(root: Path, dest_dir: Path, source_map: dict, target_class_ids: dict) -> Counter:
    """
    Copies every (image, label) pair found under root into dest_dir/images and
    dest_dir/labels, rewriting class ids in each label file according to
    source_map -> target_class_ids along the way.

    Returns a Counter of {target_class_id: instance_count} for downstream
    verification — this is what lets us catch a broken/empty download
    immediately instead of discovering it after a full training run.
    """
    images_out = dest_dir / "images"
    labels_out = dest_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    instance_counts = Counter()
    pairs_found = 0

    for images_dir, labels_dir in find_image_label_dirs(root):
        for img in tqdm(list(images_dir.iterdir()), desc=f"processing {images_dir}"):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label = labels_dir / (img.stem + ".txt")
            if not label.exists():
                continue

            lines_out = []
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                src_id = int(parts[0])
                src_label = source_map.get(src_id)
                if src_label is None:
                    continue  # unknown class in source data, drop the box
                new_id = target_class_ids[src_label]
                lines_out.append(" ".join([str(new_id)] + parts[1:]))
                instance_counts[new_id] += 1

            if not lines_out:
                continue  # background-only image, nothing to copy for our classes

            # prefix filenames with a short hash to avoid collisions across splits
            prefix = str(abs(hash(str(img))))[:8]
            shutil.copy(img, images_out / f"{prefix}_{img.name}")
            (labels_out / f"{prefix}_{label.stem}.txt").write_text("\n".join(lines_out))
            pairs_found += 1

    print(f"Copied {pairs_found} image/label pairs into {dest_dir}")
    return instance_counts


def main():
    cfg = load_config()
    dfire_dir = resolve(cfg["paths"]["dfire_dir"])
    target_class_ids = {v: k for k, v in cfg["classes"].items()}
    source_map = cfg["dfire"]["source_class_map"]
    kaggle_dataset = cfg["dfire"]["kaggle_dataset"]

    downloaded_root = download_dataset(kaggle_dataset)
    instance_counts = remap_and_copy(downloaded_root, dfire_dir, source_map, target_class_ids)

    # Hard sanity check: fail loudly instead of silently training on incomplete data.
    # This is the exact failure mode that previously went undetected — a broken
    # source (empty clone, wrong path, changed dataset layout) produced zero
    # instances for fire/smoke while the pipeline reported "success".
    expected_classes = {target_class_ids[name] for name in source_map.values()}
    missing = [cid for cid in expected_classes if instance_counts.get(cid, 0) == 0]
    if missing:
        names_by_id = {v: k for k, v in target_class_ids.items()}
        missing_names = [names_by_id[cid] for cid in missing]
        raise RuntimeError(
            f"prepare_dfire.py found ZERO instances for class(es): {missing_names}. "
            f"This almost always means the download failed or the dataset's internal "
            f"folder layout doesn't match what find_image_label_dirs() expects "
            f"(images/ + sibling labels/ folders). Inspect {downloaded_root} manually "
            f"before re-running — do not proceed to training with this data."
        )

    print(f"Done. D-Fire dataset ready at {dfire_dir} with remapped class ids: {target_class_ids}")
    print(f"Instance counts by class id: {dict(instance_counts)}")


if __name__ == "__main__":
    main()
