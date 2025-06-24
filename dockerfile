# Use the official slim Python 3.11 image
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone your repository
RUN git clone https://github.com/Pawel028/PDF-Chatbot.git .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Set a health check for the container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start Streamlit app
ENTRYPOINT ["streamlit", "run", "PDF_App.py", "--server.port=8501", "--server.address=0.0.0.0"]
