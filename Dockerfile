FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY environment.py .
COPY inference.py .
COPY server/ ./server/

RUN pip install -e .

EXPOSE 7860

# ✅ Use the 'server' console script (defined in pyproject.toml)
CMD ["server"]
