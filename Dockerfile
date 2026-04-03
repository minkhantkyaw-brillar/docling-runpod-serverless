FROM ghcr.io/docling-project/docling-serve:v1.15.0
RUN docling-tools models download layout tableformer tableformerv2 code_formula picture_classifier smolvlm granitedocling smoldocling granite_vision granite_chart_extraction rapidocr easyocr
RUN pip install --no-cache-dir runpod pyyaml

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]