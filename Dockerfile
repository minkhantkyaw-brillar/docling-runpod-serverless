FROM ghcr.io/docling-project/docling-serve:latest
RUN pip install --no-cache-dir runpod requests pyyaml

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]