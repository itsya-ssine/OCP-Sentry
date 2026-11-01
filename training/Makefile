PYTHON ?= python3

.PHONY: install prepare-coco prepare-dfire merge train evaluate export train-all clean help

help:
	@echo "Targets:"
	@echo "  make install       Install training dependencies"
	@echo "  make prepare-coco  Download COCO person subset"
	@echo "  make prepare-dfire Clone + remap D-Fire dataset"
	@echo "  make merge         Merge datasets into training/dataset/"
	@echo "  make train         Fine-tune the model (assumes data already prepared)"
	@echo "  make evaluate      Run validation on the trained model"
	@echo "  make export        Export best.pt to ONNX (+ copy into ../config/model/)"
	@echo "  make train-all     Run the FULL pipeline: install -> prepare -> merge -> train -> evaluate -> export"
	@echo "  make clean         Remove downloaded/prepared/merged data and training runs"

install:
	$(PYTHON) -m pip install -r requirements.txt --break-system-packages

prepare-coco:
	$(PYTHON) data/prepare_coco_person.py

prepare-dfire:
	$(PYTHON) data/prepare_dfire.py

merge: prepare-coco prepare-dfire
	$(PYTHON) data/merge_datasets.py

train:
	$(PYTHON) train.py

evaluate:
	$(PYTHON) evaluate.py

export:
	$(PYTHON) export.py

# One-shot target: does everything from a clean checkout to an exported ONNX model.
train-all: install merge train evaluate export
	@echo ""
	@echo "Done. Model exported to ../config/model/custom_model.onnx"

clean:
	rm -rf raw dataset runs exported
