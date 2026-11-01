"""
Exports the trained model to the format(s) configured in train_config.yaml,
and copies the ONNX output straight into config/model/ so the NVR's
inference service can pick it up with no manual file moving.

Run:
    python training/export.py [path/to/best.pt]
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from common import load_config, resolve, project_root  # noqa: E402

from ultralytics import YOLO


def main():
    cfg = load_config()
    runs_dir = resolve(cfg["paths"]["runs_dir"])
    default_weights = runs_dir / "person_fire_smoke" / "weights" / "best.pt"
    weights_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_weights

    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found — run training/train.py first.")

    export_dir = resolve(cfg["paths"]["export_dir"])
    e = cfg["export"]

    model = YOLO(str(weights_path))

    for fmt in e["formats"]:
        print(f"Exporting to {fmt}...")
        exported_path = model.export(
            format=fmt,
            imgsz=e["imgsz"],
            opset=e.get("onnx_opset", 12) if fmt == "onnx" else None,
            nms=e["nms"],
        )
        exported_path = Path(exported_path)
        dest = export_dir / exported_path.name
        shutil.copy(exported_path, dest)
        print(f"  -> {dest}")

        if fmt == "onnx":
            # drop straight into the NVR project's model slot
            nvr_model_dir = project_root() / "config" / "model"
            nvr_model_dir.mkdir(parents=True, exist_ok=True)
            nvr_dest = nvr_model_dir / "custom_model.onnx"
            shutil.copy(exported_path, nvr_dest)

            labelmap_path = nvr_model_dir / "custom_labelmap.txt"
            names = [cfg["classes"][i] for i in sorted(cfg["classes"])]
            labelmap_path.write_text("\n".join(names))

            print(f"  -> also copied to {nvr_dest}")
            print(f"  -> labelmap written to {labelmap_path}")

    print("\nExport complete.")


if __name__ == "__main__":
    main()
