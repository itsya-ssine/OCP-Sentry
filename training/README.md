# Training pipeline — person / fire / smoke model

Fine-tunes a COCO-pretrained YOLOv8 model into a 3-class detector
(`person`, `fire`, `smoke`) and exports it ready to drop into the NVR's
`config/model/` directory.

## Setup

```bash
cd training
pip install -r requirements.txt --break-system-packages
```

## Pipeline (run in order)

```bash
# 1. Pull a person-only subset of COCO (via FiftyOne), exported to YOLO format
python data/prepare_coco_person.py

# 2. Clone + remap the D-Fire dataset (fire + smoke) to this project's class ids
#    -> CHECK training/configs/train_config.yaml -> dfire.source_class_map
#       against the cloned repo's actual label convention before running.
python data/prepare_dfire.py

# 3. Merge both sources into one YOLO dataset with train/val splits + data.yaml
python data/merge_datasets.py

# 4. Fine-tune
python train.py

# 5. Check per-class accuracy (fire/smoke will likely be lower than person — expected)
python evaluate.py

# 6. Export to ONNX, auto-copied into ../config/model/custom_model.onnx
python export.py
```

## Configuration

All tunable values (epochs, batch size, dataset paths, class ids, model
size) live in `configs/train_config.yaml` — no need to edit the scripts
themselves for routine changes.

## Notes

- `train.py` starts from COCO-pretrained weights (`yolov8n.pt` by default),
  not from scratch — this transfers existing person-detection knowledge and
  speeds up convergence on the added fire/smoke classes considerably.
- Swap `train.base_model` to `yolov8s.pt` or `yolov9s.pt` in the config for
  higher accuracy at more compute cost.
- Fire/smoke detection is genuinely harder than person detection — watch
  the per-class mAP output in `evaluate.py` and expect to iterate: add more
  fire/smoke images (especially hard negatives — sunsets, orange clothing,
  steam) if those classes underperform.
- Once real camera footage is flowing through the NVR, pull false
  positive/negative clips back into `training/raw/` and re-run the merge +
  train steps — this domain-specific data will do more for accuracy than
  anything in the public datasets.
