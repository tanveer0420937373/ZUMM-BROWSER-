# Python ka base image
FROM python:3.9

# Work folder set karna
WORKDIR /app

# Files copy karna
COPY . .

# Tools install karna
RUN pip install --no-cache-dir -r requirements.txt

# Permission fix (Hugging Face ke liye zaruri hai)
RUN chmod -R 777 /app

# Server start karna (Port 7860 par)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
