# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory to /app (where the model-related files will reside)
WORKDIR /app

# Copy the requirements.txt from the model directory and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all model-related files into the /app directory
COPY . /app

# Expose the port the recommendation service will run on
EXPOSE 8001

# Run the recommendation service script when the container starts
CMD ["python", "-u", "recommendation_service.py"]