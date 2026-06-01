# Use official Python 3.10 slim image
# slim = smaller image, no unnecessary packages
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker layer caching)
# If requirements don't change, this layer is cached
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY src/ ./src/
COPY models/ ./models/

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app with uvicorn
# host 0.0.0.0 makes it accessible from outside container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
