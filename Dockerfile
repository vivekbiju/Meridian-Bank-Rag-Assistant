FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]