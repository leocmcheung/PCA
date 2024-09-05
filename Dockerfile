
FROM pytorch/pytorch:latest
# FROM python:3.10.6-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
