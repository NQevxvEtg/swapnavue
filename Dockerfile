# swapnavue/Dockerfile
# Use a CUDA-enabled base image from NVIDIA.
# Choose a version that matches your CUDA toolkit and GPU driver.
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Add labels for metadata
LABEL maintainer="NQevxvEtg"
LABEL version="1.0.0"
LABEL description="Docker image for the swapnavue backend API, featuring HTM and RFA."
LABEL org.opencontainers.image.source="https://github.com/NQevxvEtg/swapnavue"
LABEL org.opencontainers.image.licenses="MIT" 

# Set the DEBIAN_FRONTEND environment variable to noninteractive.
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.12 and its components directly from apt.
# This explicitly installs the interpreter, dev headers, and venv.
RUN apt-get update && \
    apt-get install -y wget python3.12 python3.12-dev python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
# Explicitly use python3.12 to create the venv
RUN python3.12 -m venv $VIRTUAL_ENV

# Activate the virtual environment for subsequent commands
# All 'pip' and 'python' commands from now on will operate within this venv.
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Now that the venv is active, pip is available and will install into the venv.
# No need for get-pip.py anymore, as 'python3.12 -m venv' installs pip within the venv.

# Install PyTorch with CUDA support (without torchvision/torchaudio for now)
RUN pip install --no-cache-dir torch

# Install pip-tools for pip-compile
RUN pip install --no-cache-dir pip-tools

# Copy requirements.in *just before* it's needed to generate requirements.txt
COPY requirements.in .
RUN pip-compile --output-file requirements.txt requirements.in && \
    pip install --no-cache-dir -r requirements.txt

# Copy setup.py and src/ *just before* the editable install
COPY setup.py .
COPY src src/

# Install the swapnavue package in editable mode
# This uses the copied setup.py and src directory, and installs into the venv.
RUN pip install --no-cache-dir -e .

EXPOSE 8000
# Ensure uvicorn runs from within the virtual environment
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
