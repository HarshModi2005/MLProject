FROM python:3.11-slim-bookworm

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for Playwright and PyMuPDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    wget \
    gnupg \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first to cache this layer
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser binaries
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy the rest of the application
COPY . /app/

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for Streamlit UI
EXPOSE 8501

# Command to run the application (defaults to supervised CLI mode)
ENTRYPOINT ["python", "main.py"]
CMD ["start"]
