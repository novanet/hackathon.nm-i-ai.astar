FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI + uvicorn are needed for the endpoint but not in the base requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn[standard]

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
