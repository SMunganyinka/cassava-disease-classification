FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install -r requirements-ml.txt

COPY api ./api
COPY src ./src
COPY models ./models
COPY dashboard ./dashboard

COPY results ./results

RUN mkdir -p data/uploaded

EXPOSE 8000 8501 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.getenv('PORT', '8000'); urllib.request.urlopen('http://127.0.0.1:' + port + '/health', timeout=3).read()"

CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

