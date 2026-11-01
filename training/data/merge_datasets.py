"""
Merges the prepared COCO-person subset and the remapped D-Fire dataset into
a single YOLO-format dataset with train/val splits, and writes the data.yaml
that training/train.py consumes.

Assumes prepare_coco_person.py and prepare_dfire.py have already been run.

Run:
    python training/data/merge_datasets.py
"""
import random
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

from common import load_config, resolve, project_root


def collect_pairs(root: Path):
    """Find every (image, label) pair under a directory tree with images/ and labels/ folders."""
    pairs = []
    for images_dir in root.rglob("images"):
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.exists():
            continue
        for img in images_dir.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label = labels_dir / (img.stem + ".txt")
            if label.exists():
                pairs.append((img, label))
    return pairs


def write_split(pairs, out_root: Path, split: str):
    images_out = out_root / split / "images"
    labels_out = out_root / split / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for img, label in tqdm(pairs, desc=f"writing {split}"):
        # prefix filenames with a short hash of the source path to avoid collisions
        # between datasets that might reuse filenames like "0001.jpg"
        prefix = str(abs(hash(str(img))))[:8]
        shutil.copy(img, images_out / f"{prefix}_{img.name}")
        shutil.copy(label, labels_out / f"{prefix}_{label.name}")


def main():
    cfg = load_config()
    coco_dir = resolve(cfg["paths"]["coco_person_dir"])
    dfire_dir = resolve(cfg["paths"]["dfire_dir"])
    merged_dir = resolve(cfg["paths"]["merged_dir"])
    val_fraction = cfg["split"]["val_fraction"]

    print("Collecting image/label pairs...")
    all_pairs = collect_pairs(coco_dir) + collect_pairs(dfire_dir)
    print(f"Found {len(all_pairs)} total labeled images across both sources.")

    random.seed(42)
    random.shuffle(all_pairs)
    n_val = int(len(all_pairs) * val_fraction)
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    write_split(train_pairs, merged_dir, "train")
    write_split(val_pairs, merged_dir, "val")

    names = [cfg["classes"][i] for i in sorted(cfg["classes"])]
    data_yaml = {
        "train": str((merged_dir / "train" / "images").resolve()),
        "val": str((merged_dir / "val" / "images").resolve()),
        "nc": len(names),
        "names": names,
    }
    data_yaml_path = merged_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Merged dataset written to {merged_dir}")
    print(f"  train: {len(train_pairs)} images")
    print(f"  val:   {len(val_pairs)} images")
    print(f"data.yaml written to {data_yaml_path}")


if __name__ == "__main__":
    main()
