# API service: FastAPI REST + WebSocket alert API.
FROM nvr-base:latest

WORKDIR /app
COPY config/ /app/config/
COPY services/api/main.py /app/services/api/main.py
RUN chown -R nvr:nvr /app/config /app/services

USER nvr

EXPOSE 8000
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
