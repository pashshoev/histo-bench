# Stage 1: Build Environment (using a standard Python base image - non-slim)
FROM python:3.10-bullseye

# Set working directory inside the container
WORKDIR /app

# Install git, essential runtime libraries, curl, and unzip
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libxrender1 \
    libxext6 \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install gdc_client
# Using curl to download and then unzip. Placing it in /usr/local/bin for global access.
RUN curl -LO https://gdc.cancer.gov/system/files/public/file/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip && \
    unzip gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip && \
    unzip gdc-client_2.3_Ubuntu_x64.zip && \
    mv gdc-client /usr/local/bin/ && \
    rm -rf gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip \
           gdc-client_2.3_Ubuntu_x64.zip \
           gdc-client

# Copy your requirements.txt file
COPY requirements.txt .

# Install Python dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application directory into the container (assuming ui/1_Home.py is in your current build context)
COPY . .

# Set PYTHONPATH environment variable for the application
ENV PYTHONPATH=/app

# Expose any ports your application uses (for Streamlit)
EXPOSE 8501

# Command to run your Streamlit application when the container starts
CMD ["python", "-m", "streamlit", "run", "ui/1_Home.py"]