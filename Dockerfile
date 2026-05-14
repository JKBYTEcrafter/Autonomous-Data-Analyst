FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p datasets reports_output app/models

# Make startup script executable
RUN chmod +x start.sh

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["./start.sh"]
