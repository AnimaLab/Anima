# LongMemEval benchmark runner image
# Build from benchmarks/longmemeval/:
#   docker build -t bench-longmemeval .

FROM python:3.12-slim

WORKDIR /bench

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py .env ./
COPY data/ data/

ENTRYPOINT ["python", "run.py"]
