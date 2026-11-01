"""Quick-start helper: exports a stock COCO-pretrained YOLOv8 model straight
into config/model/, so you can test the full ingest -> inference -> events
-> API -> frontend pipeline (e.g. against a webcam) before training the
custom person/fire/smoke model.

The stock model detects all 80 COCO classes, including "person" — but has
no "fire" or "smoke" classes, since those aren't in COCO. Set
`detection.classes_of_interest: ["person"]` in config/settings.yaml while
using it, or you'll just never see fire/smoke alerts (expected, not a bug).

Run:
    pip install ultralytics --break-system-packages
    python training/export_pretrained.py            # uses yolov8n.pt
    python training/export_pretrained.py yolov8s.pt  # or a bigger variant
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
    print(f"Loading {weights} (auto-downloads on first run)...")
    model = YOLO(weights)

    print("Exporting to ONNX (NMS baked in, matching training/export.py's convention)...")
    exported_path = Path(model.export(format="onnx", imgsz=640, opset=12, nms=True))

    root = Path(__file__).resolve().parent.parent
    dest_dir = root / "config" / "model"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / "custom_model.onnx"
    shutil.copy(exported_path, dest)

    names = [model.model.names[i] for i in sorted(model.model.names)]
    labelmap_path = dest_dir / "custom_labelmap.txt"
    labelmap_path.write_text("\n".join(names))

    print(f"\n  -> {dest}")
    print(f"  -> {labelmap_path} ({len(names)} COCO classes)")
    print(
        "\nThis is the stock COCO model, not your fine-tuned one — it has no "
        "fire/smoke classes. Set:\n"
        '  detection.classes_of_interest: ["person"]\n'
        "in config/settings.yaml until you run the real training pipeline "
        "(training/train.py + training/export.py), which overwrites these "
        "same two files with your custom 3-class model."
    )


if __name__ == "__main__":
    main()
