# --- Stage 1: The "Builder" ---
# We give it a name "builder" so we can refer to it later
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install CPU-specific torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY src/main/requirements_cpu.txt .
RUN pip install --no-cache-dir -r requirements_cpu.txt

# --- This is the key part ---
# 1. Copy EVERYTHING in (the big 11GB+ copy)
COPY . .

# 2. Now run your `rm` commands to clean up this stage
RUN rm .gitmodules
RUN rm README.md
RUN rm -rf src/main/python/L3/ObjectDetection/detrex_ITA
RUN rm -rf src/main/python/L3/ObjectDetection/ObjectDetectionDetrex.py
RUN rm -rf checkpoints
# --- End of key part ---


# --- Stage 2: The "Final" Image ---
# This is a new, completely clean image
FROM python:3.10-slim

WORKDIR /app

# Copy system dependencies from the builder stage
COPY --from=builder /usr/lib /usr/lib
COPY --from=builder /usr/include /usr/include
COPY --from=builder /usr/bin/gdal* /usr/bin/

# Set environment variables again
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# --- THIS IS THE NEW LINE ---
# Copy the Python executables (like streamlit, pip, etc.)
COPY --from=builder /usr/local/bin /usr/local/bin
# --- END OF NEW LINE ---

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy your cleaned-up application code from the builder stage
COPY --from=builder /app /app

# Run the app
CMD ["streamlit", "run", "src/main/python/main_streamlit_v2.py"]


