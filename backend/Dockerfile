FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ app/
# Bake the trial corpus into the image so Cloud Run doesn't need a volume mount.
# This is the full 10k+ trial dataset that config.py points at by default;
# update both this line and config.csv_path when pulling a fresh CSV.
COPY data/breast_cancer_trials_full_2026-04-14.csv data/
# Bake the precomputed Voyage embeddings cache so startup is ~3 seconds instead
# of ~10 minutes. Cache is keyed on sha256(model + corpus); regenerated on miss.
COPY data/embeddings_cache/ data/embeddings_cache/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
