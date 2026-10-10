FROM python:3.14-slim

# Keep project files under /app inside the container
WORKDIR /app

# Install the model and API packages
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the API code and saved model into the image
COPY backend/ ./backend/
COPY trained_model/saved_models/neurox-v1/ ./trained_model/saved_models/neurox-v1/

# Copy the three request schemas loaded by the API
COPY ["API specs and schemas/predict_request.schema.json", "API specs and schemas/batch_request.schema.json", "API specs and schemas/batch_job_request.schema.json", "/app/API specs and schemas/"]

# Show logs immediately in Docker
ENV PYTHONUNBUFFERED=1

# Document the port used by the API
EXPOSE 8000

# Listen on all container network interfaces
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]