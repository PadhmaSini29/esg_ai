# Use the official Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port your app runs on (if using Flask, often 5000)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
