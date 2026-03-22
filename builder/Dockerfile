FROM python:3.11-slim

RUN groupadd -r agent && useradd -r -g agent -m agent

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_code/inference ./inference
COPY app_code/agent ./agent
COPY app_code/admin ./admin
COPY app_code/data_pipeline ./data_pipeline

COPY data ./data
COPY models ./models
COPY config.yaml .

RUN mkdir -p /app/data/incoming /app/data/docs && \
    chown -R agent:agent /app

USER agent

EXPOSE 8000 8001

CMD ["sh", "-c", "uvicorn inference.server:app --host 0.0.0.0 --port 8000 & uvicorn admin.app:admin_app --host 0.0.0.0 --port 8001 & wait"]
