# Use Python 3.11 (stable for Streamlit and LangChain)
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements first (leverage Docker cache)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Copy Google Cloud key to container
COPY gcs-key.json /app/gcs-key.json

# Set environment variable for Google SDK
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/gcs-key.json"
