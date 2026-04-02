FROM ghcr.io/docling-project/docling-serve:v1.15.0
RUN docling-tools models download --all
RUN pip install --no-cache-dir runpod pyyaml

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]