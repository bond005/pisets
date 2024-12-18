FROM python:3.10-alpine

RUN pip install --no-cache-dir gradio==5.9.1 requests==2.32.3 pandas==2.2.3

COPY web_client.py /app/web_client.py

EXPOSE 80
CMD ["python3.10", "/app/web_client.py"]