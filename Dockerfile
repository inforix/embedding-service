# Use NVIDIA CUDA base image
FROM harbor.shmtu.edu.cn/nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . /app/

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
