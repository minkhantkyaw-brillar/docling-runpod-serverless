FROM ghcr.io/docling-project/docling-serve-cu128:v1.35.0

USER 0

RUN dnf install -y \
    tesseract \
    tesseract-devel \
    tesseract-langpack-eng \
    tesseract-osd \
    leptonica-devel \
    && dnf clean all

ENV TESSDATA_PREFIX=/usr/share/tesseract/tessdata/
ENV DOCLING_SERVE_ALLOW_CUSTOM_OCR_CONFIG=true \
    DOCLING_SERVE_DEBUG_ERROR_DETAILS=true \
    DOCLING_DEVICE=cuda

RUN echo "Set TESSDATA_PREFIX=${TESSDATA_PREFIX}"

RUN docling-tools models download layout tableformer tableformerv2 code_formula picture_classifier smolvlm granitedocling smoldocling granite_vision granite_chart_extraction rapidocr easyocr
RUN python -m pip install --no-cache-dir runpod pyyaml docling[easyocr,rapidocr] tesserocr onnxruntime

USER 1001

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
