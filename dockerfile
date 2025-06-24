# Use the official Python 3.11 slim image as the base
FROM python:3.11-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "PDF_App.py", "--server.port=8501", "--server.address=0.0.0.0"]