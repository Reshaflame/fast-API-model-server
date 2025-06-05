# CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV RAY_memory=auto
ENV RAY_cpu=auto
ENV RAY_gpu=auto
ENV PYTHONPATH="/app"
ENV NUMBA_CUDA_DRIVER=/usr/local/cuda/compat/libcuda.so.1

# CUDA paths
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    python --version

# Install cuDF + RAPIDS (CUDA 12.1)
RUN pip install --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==23.12 dask-cudf-cu12==23.12 --prefer-binary --no-cache-dir

RUN pip install --no-cache-dir numba==0.57.1

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir Flask gdown && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Copy source code
COPY ./app /app/app
COPY ./app/data /app/data
COPY ./app/models /app/models


# Set working directory
WORKDIR /app

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
