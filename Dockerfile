ARG BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ROOT=/models \
    DEVICE=cuda \
    MAX_BATCH_SIZE=8 \
    MAX_BATCH_WAIT_MS=2 \
    MAX_QUEUE_SIZE=256 \
    MODEL_CACHE_SIZE=2

WORKDIR /app

COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY . .

EXPOSE 8998
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8998/health', timeout=3)"

CMD ["python", "-m", "serving.app", "--host", "0.0.0.0", "--port", "8998", "--workers", "1"]
