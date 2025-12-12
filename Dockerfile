FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY flows/ ./flows/

RUN mkdir -p /secrets
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "flows.opensky_ingest_flow"]
