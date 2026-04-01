FROM python:3.11-slim

WORKDIR /app

# 🔥 ADD THIS BLOCK (IMPORTANT FIX)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port (Railway ignores but fine)
EXPOSE 8000

# Run API
CMD ["python", "scripts/run_api.py"]
