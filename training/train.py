"""
Fine-tunes a COCO-pretrained YOLO model on the merged person/fire/smoke
dataset produced by data/merge_datasets.py.

Run:
    python training/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from common import load_config, resolve, project_root  # noqa: E402

from ultralytics import YOLO


def main():
    cfg = load_config()
    data_yaml = resolve(cfg["paths"]["merged_dir"]) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found — run data/prepare_coco_person.py, "
            f"data/prepare_dfire.py, then data/merge_datasets.py first."
        )

    runs_dir = resolve(cfg["paths"]["runs_dir"])
    t = cfg["train"]

    model = YOLO(t["base_model"])  # starts from COCO-pretrained weights
    model.train(
        data=str(data_yaml),
        imgsz=t["imgsz"],
        epochs=t["epochs"],
        batch=t["batch"],
        patience=t["patience"],
        device=t["device"],
        project=str(runs_dir),
        name="person_fire_smoke",
    )

    best = runs_dir / "person_fire_smoke" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")
    print("Next: python training/evaluate.py, then python training/export.py")


if __name__ == "__main__":
    main()
