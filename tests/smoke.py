"""Integration checks against the actual bundled Docling Serve (CPU compatible)."""
import base64
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import handler

handler.configure_logging()
handler.verify_ocr()
source = {"kind": "file", "filename": "smoke.html", "base64_string": base64.b64encode(
    b"<html><body><h1>Worker smoke test</h1><p>Docling conversion works.</p></body></html>"
).decode()}

def run(payload):
    return handler.handler({"id": "smoke", "input": payload})

try:
    payload = {"sources": [source], "options": {"to_formats": ["md", "json"]}}
    result = run(payload)
    assert "error" not in result, result
    assert "Worker smoke test" in json.dumps(result), result
    zipped = run({**payload, "target": {"kind": "zip"}})
    with ZipFile(BytesIO(base64.b64decode(zipped["base64_string"]))) as archive:
        assert any(name.endswith(".md") for name in archive.namelist()), archive.namelist()
    invalid = run({**payload, "options": {"ocr_engine": "not-an-ocr-engine"}})
    assert "error" in invalid, invalid
    assert "error" in run({"sources": []})
    assert "error" in run({**payload, "target": {"kind": "presigned_url"}})
    # A failed request must not poison subsequent jobs.
    assert "error" not in run(payload)
    print("PASS: imports, inbody conversion, ZIP export, validation, worker reuse", flush=True)
finally:
    handler.service.stop()
