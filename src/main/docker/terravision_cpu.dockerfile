# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

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

# Set environment variable to help pip find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

COPY . .

RUN pip install --upgrade pip setuptools wheel

RUN pip install torch torchvision torchaudio

# Install requirements
RUN pip install --no-cache-dir -r src/main/requirements_cpu.txt

# Run the app
CMD ["streamlit", "run", "src/main/python/main_streamlit.py"]
