FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install FFmpeg development libraries, build tools, and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libsndfile1 \
    libgl1 \
    libxkbcommon0 \
    libglib2.0-0 \
    ffmpeg \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    netcat-openbsd \
    libopus-dev \
    libvpx-dev \
    libsrtp2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    --default-timeout=100 \
    --break-system-packages \
    -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check (using netcat for a robust check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nc -z localhost 8000 || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]