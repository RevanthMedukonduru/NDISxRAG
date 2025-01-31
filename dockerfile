FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (for Docker build cache)
COPY requirements.txt /app/

# Update APT, install netcat, then clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       netcat-openbsd \
       pkg-config \
       gcc \
       python3-dev \
       default-libmysqlclient-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . /app/

EXPOSE 8000
EXPOSE 8501

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
