FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Train inside the image so the container is ready to serve on start.
RUN python -m bcd.train

EXPOSE 8000
CMD ["uvicorn", "bcd.api:app", "--host", "0.0.0.0", "--port", "8000"]
