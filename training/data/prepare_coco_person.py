"""
Downloads a person-only subset of COCO via FiftyOne and exports it to plain
YOLO format (images/ + labels/ with one .txt per image), remapped to class
id 0 ("person") to match training/configs/train_config.yaml.

Run:
    python training/data/prepare_coco_person.py
"""
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from tqdm import tqdm

from common import load_config, resolve


def export_split(dataset, out_dir: Path, class_id: int):
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(dataset, desc=f"exporting to {out_dir.name}"):
        img_path = Path(sample.filepath)
        dst_img = images_dir / img_path.name
        shutil.copy(img_path, dst_img)

        w, h = sample.metadata.width, sample.metadata.height
        lines = []
        detections = sample.ground_truth.detections if sample.ground_truth else []
        for det in detections:
            if det.label != "person":
                continue
            # FiftyOne stores bounding_box as [x, y, w, h] normalized top-left origin
            x, y, bw, bh = det.bounding_box
            cx = x + bw / 2
            cy = y + bh / 2
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = labels_dir / (img_path.stem + ".txt")
        label_path.write_text("\n".join(lines))


def main():
    cfg = load_config()
    person_class_id = next(k for k, v in cfg["classes"].items() if v == "person")
    out_root = resolve(cfg["paths"]["coco_person_dir"])
    max_samples = cfg["coco"]["max_samples_per_split"]

    for split in cfg["coco"]["splits"]:
        print(f"Downloading COCO-2017 [{split}] filtered to 'person' (max {max_samples} samples)...")
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            label_types=["detections"],
            classes=["person"],
            max_samples=max_samples,
            dataset_name=f"coco_person_{split}",
        )
        export_split(dataset, out_root / split, person_class_id)

    print(f"Done. Person subset written to {out_root}")


if __name__ == "__main__":
    main()
