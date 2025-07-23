# 1. Use CUDA-Toolkit “devel” image so cuda_runtime_api.h is available
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 2. Install Python 3.10, build tools, GDAL system libs, and other native deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3.10 python3.10-dev python3-pip python3-distutils \
      build-essential ninja-build git \
      libjpeg-dev zlib1g-dev libpng-dev \
      libgl1-mesa-glx libglib2.0-dev \
      libsm6 libxext6 libxrender-dev \
      # --- GDAL native library & headers for Python bindings ---
      libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# 3. Point Python & pip to 3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3    1

# 4. Expose CUDA include & lib paths, and GDAL config
ENV CUDA_HOME=/usr/local/cuda \
    CPATH=$CUDA_HOME/include:$CPATH \
    LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH \
    # Tell pip where gdal-config lives
    GDAL_CONFIG=/usr/bin/gdal-config \
    # Ensure include path picked up
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# 11. Enable CUDA for Detrex extensions
ENV USE_CUDA=1

# 5. Copy your project files
COPY . .

# 6. Upgrade pip/build tools and install Cython & numpy first
RUN pip install --upgrade pip setuptools wheel \
 && pip install Cython numpy

# 7. Install a CUDA-enabled PyTorch wheel
RUN pip install torch torchvision torchaudio

# 8. Install GroundingDINO & Grounded-SAM-2
RUN pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git \
 && pip install src/main/python/L3/ObjectDetection/Grounded-SAM-2 \
 && pip install src/main/python/L3/ObjectDetection/Grounded-SAM-2[demo]

# 9. Install the rest of your GPU requirements (including rasterio, gdal-python, etc)

RUN pip install -r src/main/requirements_gpu.txt

CMD ["streamlit", "run", "src/main/python/main_streamlit.py"]
