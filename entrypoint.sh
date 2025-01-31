#!/bin/bash

set -e  # Exit on error

# Wait for Redis to be ready (port 6379)
echo "Waiting for Redis to be ready..."
until nc -z -v -w30 redis 6379; do
  echo "Waiting for Redis connection..."
  sleep 1
done
echo "Redis is ready!"

# Wait for MySQL to be ready (port 3306)
echo "Waiting for MySQL to be ready..."
until nc -z -v -w30 mysql 3306; do
  echo "Waiting for MySQL connection..."
  sleep 1
done
echo "MySQL is ready!"

# Run Django migrations (assuming your DATABASES setting points to MySQL or another DB)
echo "Running Django migrations..."
python manage.py migrate --noinput || echo "No migrations or DB not configured."

echo "Starting Django development server..."
exec "$@"