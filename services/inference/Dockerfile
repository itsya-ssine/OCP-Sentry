# Inference service: batched ONNX (onnxruntime) inference worker.
FROM nvr-base:latest

WORKDIR /app
COPY config/ /app/config/
COPY services/inference/inference_worker.py /app/services/inference/inference_worker.py
RUN chown -R nvr:nvr /app/config /app/services

USER nvr

CMD ["python", "services/inference/inference_worker.py"]
