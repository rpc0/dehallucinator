# # Use an official NVIDIA CUDA runtime as a parent image
# FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu20.04

#FROM ubuntu/python:3.12-24.04_stable
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install PyTorch (GPU version)
# RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==0.13.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install LlamaIndex
RUN pip install llama-index

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install LlamaIndex-LLMS-Ollama
RUN pip install llama-index-llms-ollama 

# Install Haystack
RUN pip install farm-haystack 

# Install Langchain
RUN pip install langchain langchain-community langchain-openai langchain-huggingface

# Install OpenAI package
RUN pip install openai gradio

# Install other requirements
# RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Gradio uses
EXPOSE 7860

# Set environment variable to allow Gradio to listen on all interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Copy your application code
COPY . .

# Command to run your application
CMD ["python", "app.py"]	

# Build the Docker image
# docker buildx build --no-cache -t delucinator:latest

# Run the Docker container
# docker run -it delucinator:latest