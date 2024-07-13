# Use the official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Download the model (ensure this URL is correct)
RUN python -c "import gdown; gdown.download('https://drive.google.com/file/d/10rJ8GQ-_5eq2uGh5hEk5Tio7tZ6IbHpa/view?usp=sharing', './fine_tuned_model/model.safetensors', quiet=False)"

# Specify the command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "Flask_Test:app"]
