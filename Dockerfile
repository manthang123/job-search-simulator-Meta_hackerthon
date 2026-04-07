FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment
COPY environment.py .
COPY openenv.yaml .

# Expose port
EXPOSE 7860

# Run OpenEnv server
CMD ["python", "-c", "from openenv import serve; from environment import JobSearchSimulator; serve(JobSearchSimulator)"]