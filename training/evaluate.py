"""
Runs validation on the trained model and prints per-class metrics — check
fire/smoke mAP specifically, since those classes are harder than person and
usually need more data before evaluate.py shows them near person's score.

Run:
    python training/evaluate.py [path/to/best.pt]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from common import load_config, resolve  # noqa: E402

from ultralytics import YOLO


def main():
    cfg = load_config()
    runs_dir = resolve(cfg["paths"]["runs_dir"])
    default_weights = runs_dir / "person_fire_smoke" / "weights" / "best.pt"
    weights_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_weights

    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found — run training/train.py first.")

    data_yaml = resolve(cfg["paths"]["merged_dir"]) / "data.yaml"

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), imgsz=cfg["train"]["imgsz"])

    names = cfg["classes"]
    print("\nPer-class mAP50-95:")
    for i, class_name in sorted(names.items()):
        try:
            print(f"  {class_name:10s}: {metrics.box.maps[i]:.3f}")
        except (IndexError, AttributeError):
            print(f"  {class_name:10s}: n/a")

    print(f"\nOverall mAP50:    {metrics.box.map50:.3f}")
    print(f"Overall mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
