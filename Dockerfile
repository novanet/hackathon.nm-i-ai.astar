FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard]

COPY astar/ astar/
COPY main.py .
COPY data/spatial_model.pkl data/spatial_model.pkl

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
