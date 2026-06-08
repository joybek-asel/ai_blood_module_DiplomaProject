FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Train the model at build time — generates /app/models/ inside the image
RUN python src/recommendation/train_model.py

CMD ["uvicorn", "src.recommendation.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
