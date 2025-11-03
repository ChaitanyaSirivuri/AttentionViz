# Use an official Python runtime as a parent image
# Using slim to keep it smaller, but full might be safer for complex ML dependencies if needed.
# 3.10 is a good stable version for ML libraries.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# git is often needed for installing dependencies from git or for some python packages
# libgl1-mesa-glx is needed for opencv-python (if used by any sub-dependency) or similar image libs
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to keep image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
# Prevents Python from writing pyc files to disc (equivalent to python -B option)
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr (equivalent to python -u option)
ENV PYTHONUNBUFFERED 1

# Run streamlit when the container launches
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
