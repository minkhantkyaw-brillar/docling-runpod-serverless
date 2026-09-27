#!/usr/bin/env python3
"""Live RunPod integration matrix. --list loads configuration but never reads source contents or calls the API."""
import argparse
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
FORMATS = {"md": "md_content", "json": "json_content", "html": "html_content",
           "text": "text_content", "yaml": "yaml_content", "doctags": "doctags_content"}

@dataclass
class Case:
    name: str
    source: Path
    options: dict = field(default_factory=dict)
    raster: bool = False
    target: str = "inbody"
    check: str = "text"
    expected_page: int | None = None


def source_directory(value):
    if not value.strip():
        raise ValueError("Set TEST_SOURCE_DIR in .env or the environment; there is no default source directory")
    folder = Path(value.strip()).expanduser()
    if not folder.is_absolute():
        folder = ROOT / folder
    if not folder.is_dir():
        raise ValueError("TEST_SOURCE_DIR must point to an existing directory")
    return folder.resolve()


def cases_for(folder, extended=False):
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".xlsx"})
    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
    if not pdfs:
        raise ValueError("Source directory must contain at least one PDF")
    cases = [Case(f"baseline-{i}", p, {"to_formats": ["md", "json"]}) for i, p in enumerate(files)]
    for i, pdf in enumerate(pdfs):
        for engine in ("tesseract", "rapidocr", "easyocr", "tesseract_cli"):
            cases.append(Case(f"ocr-{engine}-{i}", pdf,
                {"to_formats": ["md", "json"], "do_ocr": True, "force_ocr": True,
                 "ocr_engine": engine, "ocr_lang": ["en"], "do_table_structure": False}, raster=True))
    pdf = pdfs[0]
    cases += [
        Case("ocr-disabled-control", pdf, {"do_ocr": False, "do_table_structure": False,
             "to_formats": ["md"]}, raster=True, check="empty"),
        Case("page-one", pdf, {"page_range": [1, 1], "to_formats": ["md", "json"]}, expected_page=1),
        Case("page-two", pdf, {"page_range": [2, 2], "to_formats": ["md", "json"]}, expected_page=2),
        Case("tables-fast", pdf, {"page_range": [1, 1], "table_mode": "fast", "table_cell_matching": False}),
        Case("tables-disabled", pdf, {"page_range": [1, 1], "do_table_structure": False,
             "to_formats": ["md", "json"]}, check="no_tables"),
        Case("pdfium", pdf, {"page_range": [1, 1], "pdf_backend": "pypdfium2"}),
        Case("text-without-ocr", pdf, {"page_range": [1, 1], "do_ocr": False}, check="exports"),
        Case("exports", pdf, {"page_range": [1, 1], "to_formats": list(FORMATS)}),
        Case("embedded-page-image", pdf, {"page_range": [1, 1], "to_formats": ["json"],
             "image_export_mode": "embedded", "include_page_images": True,
             "include_images": True, "images_scale": 1.0}, check="image"),
        Case("referenced-zip", pdf, {"page_range": [1, 1], "to_formats": ["md", "json"],
             "image_export_mode": "referenced", "include_page_images": True}, target="zip", check="zip"),
        Case("invalid-ocr", pdf, {"ocr_engine": "INVALID_BACKEND"}, check="rejected"),
        Case("invalid-format", pdf, {"to_formats": ["INVALID_FORMAT"]}, check="rejected"),
        Case("invalid-page-range", pdf, {"page_range": [0, 0]}, check="rejected"),
        Case("reuse-after-errors", pdf, {"page_range": [1, 1]}),
    ]
    if extended:
        for option in ("do_code_enrichment", "do_formula_enrichment", "do_picture_classification",
                       "do_picture_description", "do_chart_extraction", "do_pdf_heading_hierarchy"):
            cases.append(Case(option, pdf, {"page_range": [1, 1], option: True}))
        cases.append(Case("vlm-granite", pdf, {"page_range": [1, 1], "pipeline": "vlm",
                                            "vlm_pipeline_preset": "granite_docling"}))
    return cases


def source_for(case):
    content = case.source.read_bytes()
    name = case.source.name
    if case.raster:
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError("Install tests/requirements.txt to render OCR fixtures") from exc
        with pymupdf.open(case.source) as document:
            content = document[0].get_pixmap(dpi=150, alpha=False).tobytes("png")
        name = case.source.stem + "-page1.png"
    return {"kind": "file", "filename": name, "base64_string": base64.b64encode(content).decode()}


def endpoint_url(value):
    value = value.strip().rstrip("/")
    if value.endswith(("/run", "/runsync")):
        value = value.rsplit("/", 1)[0]
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
            or parsed.query or parsed.fragment or not re.fullmatch(r"/v2/[^/]+", parsed.path)):
        raise ValueError("RUNPOD_ENDPOINT_URL must be https://api.runpod.ai/v2/ENDPOINT_ID")
    return value


class Client:
    def __init__(self, endpoint, key, timeout=1800, interval=5):
        self.endpoint, self.key = endpoint, key
        self.timeout, self.interval = timeout, interval
        self.active_job = None

    def request(self, method, path, data=None, timeout=60):
        body = json.dumps(data).encode() if data is not None else None
        if body and len(body) > 9_500_000:
            raise ValueError("Encoded request exceeds conservative /run payload budget (9.5 MB)")
        request = Request(self.endpoint + path, data=body, method=method,
                          headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            code = exc.code
            exc.close()
            raise RuntimeError(f"RunPod HTTP {code} on {method} {path.split('/')[1]}") from None
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"RunPod {method} connection failed ({type(exc).__name__}); submissions are not retried") from None

    def run(self, payload, submitted):
        # Do not retry POST /run: an ambiguous response could create duplicate paid jobs.
        result = self.request("POST", "/run", {"input": payload})
        job_id = result.get("id")
        if not job_id:
            raise RuntimeError("RunPod submission returned no job ID")
        self.active_job = str(job_id)
        submitted(self.active_job)
        deadline = time.monotonic() + self.timeout
        previous = None
        poll_errors = 0
        while True:
            status = result.get("status")
            if status != previous:
                print(f"  job={job_id} status={status}", flush=True)
                previous = status
            if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                self.active_job = None
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Job {job_id} exceeded client wait timeout")
            time.sleep(min(self.interval, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                continue
            try:
                result = self.request("GET", "/status/" + quote(str(job_id), safe=""), timeout=min(60, remaining))
                poll_errors = 0
            except RuntimeError as exc:
                if not re.search(r"HTTP (429|500|502|503|504)\b|GET connection failed", str(exc)):
                    raise
                poll_errors += 1
                if poll_errors > 3:
                    raise
                print(f"  Temporary status error; retrying existing job ({poll_errors}/3)", flush=True)

    def cancel(self):
        if self.active_job:
            job_id = self.active_job
            try:
                self.request("POST", "/cancel/" + quote(job_id, safe=""))
                print(f"  Cancellation requested for {job_id}", flush=True)
                self.active_job = None
            except Exception:
                print(f"  Cancellation unconfirmed; check job {job_id} in RunPod", flush=True)


def values(value, key):
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for child in value.values():
            yield from values(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from values(child, key)


def validate(case, response):
    output = response.get("output")
    failed = response.get("status") == "FAILED" or isinstance(output, dict) and bool(output.get("error"))
    if case.check == "rejected":
        # A crash, timeout, or generic worker failure does not prove validation worked.
        message = json.dumps(response)
        if not failed or not re.search(r"HTTP 4(?:00|22)|validation|invalid.*(?:ocr|format|page)", message, re.I):
            raise AssertionError("Expected an explicit input-validation failure")
        return {"validation_rejected": True}
    if response.get("status") != "COMPLETED" or failed or not isinstance(output, dict):
        raise AssertionError("Job did not complete with a successful object output; inspect saved response")
    statuses = list(values(output, "status"))
    if any(s in ("failure", "partial_success", "skipped") for s in statuses):
        raise AssertionError("Docling reported failure, partial success, or skipped conversion")
    if any(bool(e) for e in values(output, "errors")):
        raise AssertionError("Docling returned conversion errors")
    if case.check == "zip":
        with ZipFile(BytesIO(base64.b64decode(output["base64_string"], validate=True))) as archive:
            names = archive.namelist()
            if archive.testzip() or not any(n.endswith(".md") for n in names) or not any(n.endswith(".json") for n in names):
                raise AssertionError("ZIP missing valid Markdown/JSON exports")
            if not any(n.lower().endswith((".png", ".jpg", ".jpeg")) for n in names):
                raise AssertionError("Referenced ZIP is missing image assets")
        return {"archive_files": names}
    for fmt in case.options.get("to_formats", ["md"]):
        entries = list(values(output, FORMATS[fmt]))
        if not entries or all(v is None for v in entries):
            raise AssertionError(f"Missing {fmt} export")
    texts = list(values(output, "md_content")) + list(values(output, "text_content"))
    # Only document text, not filenames, schemas, or base64 images, counts as OCR evidence.
    for doc in values(output, "json_content"):
        if isinstance(doc, dict):
            texts += [t.get("text", "") for t in doc.get("texts", [])]
    text = "\n".join(t for t in texts if isinstance(t, str))
    text = re.sub(r"<!--.*?-->|!\[[^\]]*\]\([^)]*\)", "", text, flags=re.S)
    letters = sum(c.isalnum() for c in text)
    if case.check in {"text", "no_tables"} and letters < 20:
        raise AssertionError(f"Too little extracted text ({letters} alphanumeric characters)")
    if case.check == "empty" and letters:
        raise AssertionError("OCR-disabled raster control unexpectedly produced text")
    if case.check == "no_tables":
        for doc in values(output, "json_content"):
            if isinstance(doc, dict) and doc.get("tables"):
                raise AssertionError("Tables were returned with do_table_structure=false")
    if case.expected_page is not None:
        pages = set()
        for doc in values(output, "json_content"):
            if isinstance(doc, dict):
                pages.update(int(k) for k in doc.get("pages", {}))
        if pages != {case.expected_page}:
            raise AssertionError(f"Expected only page {case.expected_page}, got {sorted(pages)}")
    if case.check == "image" and "data:image/" not in json.dumps(output):
        raise AssertionError("Embedded page-image data missing")
    return {"text_characters": letters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tests/results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--list", action="store_true", help="List cases using TEST_SOURCE_DIR without reading document contents or submitting jobs")
    parser.add_argument("--match", default="", help="Run case names containing this string")
    parser.add_argument("--smoke", action="store_true", help="Only the smallest PDF with to_formats=[md]")
    parser.add_argument("--extended", action="store_true", help="Add model-heavy enrichment and VLM checks")
    parser.add_argument("--timeout", type=float, default=1800, help="Per-job wait including queue time, seconds")
    parser.add_argument("--poll-interval", type=float, default=5)
    args = parser.parse_args()
    if min(args.timeout, args.poll_interval) <= 0:
        parser.error("Timeout and polling interval must be positive")
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)
    folder = source_directory(os.getenv("TEST_SOURCE_DIR", ""))
    cases = [c for c in cases_for(folder, args.extended) if args.match in c.name]
    if args.smoke:
        pdfs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
        cases = [Case("smoke-markdown", min(pdfs, key=lambda p: p.stat().st_size), {"to_formats": ["md"]})]
    if not cases:
        parser.error("No matching cases")
    if args.list:
        for case in cases:
            print(f"{case.name}: {case.source.name} | raster={case.raster} | {json.dumps(case.options)}")
        print(f"{len(cases)} cases; no source contents read, no API calls")
        return 0
    endpoint = endpoint_url(os.getenv("RUNPOD_ENDPOINT_URL", ""))
    key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not key:
        parser.error("Set RUNPOD_API_KEY in .env or the environment")
    client = Client(endpoint, key, args.timeout, args.poll_interval)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"cases": [], "note": "Smoke/behavior checks; OCR accuracy is not scored against ground truth."}
    def save():
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    try:
        for case in cases:
            started = time.monotonic()
            row = {"name": case.name, "source": case.source.name, "options": case.options,
                   "raster": case.raster, "target": case.target, "status": "RUNNING"}
            report["cases"].append(row)
            print(f"RUN {case.name} ({case.source.name})", flush=True)
            def submitted(job_id):
                row["job_id"] = job_id
                save()
            try:
                payload = {"sources": [source_for(case)], "options": case.options, "target": {"kind": case.target}}
                response = client.run(payload, submitted)
                (args.output_dir / f"{case.name}.json").write_text(json.dumps(response, indent=2))
                row.update(validate(case, response))
                row["status"] = "PASS"
            except Exception as exc:
                row.update(status="FAIL", reason=str(exc))
                client.cancel()
            row["elapsed_seconds"] = round(time.monotonic() - started, 2)
            save()
            print(f"{row['status']} {case.name}: {row.get('reason', 'checks passed')}", flush=True)
            if client.active_job:
                break  # Avoid piling up jobs when cancellation could not be confirmed.
    except KeyboardInterrupt:
        client.cancel()
        row.update(status="INTERRUPTED")
        save()
        return 130
    failures = sum(row["status"] != "PASS" for row in report["cases"])
    print(f"{len(report['cases']) - failures} passed, {failures} failed. Report: {args.output_dir / 'report.json'}")
    return int(bool(failures))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
