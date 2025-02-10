FROM python:3.11-slim

ENV PROJECT_DIR model_in_image
WORKDIR /${PROJECT_DIR}
ADD requirements.txt /${PROJECT_DIR}/
RUN apt-get -y update && \
    apt-get -y install \
    apt-utils \
    gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt 

COPY src /${PROJECT_DIR}/src/
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
