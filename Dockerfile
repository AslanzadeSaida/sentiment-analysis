# Use an official lightweight Python image
FROM python:3.9-slim

# Create working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# RUN python -m src.train_model


COPY . .

EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--access-log", "--log-level", "info"]

