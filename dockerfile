# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for many Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone your repository
RUN git clone https://github.com/Pawel028/PDF-Chatbot.git .

# COPY . .
# Install Python dependencies
RUN pip install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Set a health check for the container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start Streamlit app
# ENTRYPOINT ["streamlit", "run", "PDF_App.py", "--server.port=8501", "--server.address=0.0.0.0"]
ENTRYPOINT ["streamlit", "run", "PDF_App.py"]
