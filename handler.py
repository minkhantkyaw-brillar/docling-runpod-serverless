"""RunPod JSON adapter for the bundled Docling Serve asynchronous conversion API."""

import atexit
import base64
import importlib
import importlib.metadata
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

logger = logging.getLogger("docling_runpod")
_TERMINAL = {"success", "partial_success", "failure", "skipped"}


def configure_logging():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )


def verify_ocr():
    """Fail startup early if the promised OCR backends cannot be imported."""
    for name in ("tesserocr", "rapidocr", "easyocr"):
        module = importlib.import_module(name)
        logger.info("OCR backend=%s version=%s import=ok", name, importlib.metadata.version(name))
        if name == "tesserocr":
            path, languages = module.get_languages()
            logger.info("Tesseract tessdata=%s languages=%s", path, languages)
            if "eng" not in languages:
                raise RuntimeError("Tesseract English language data is missing")
    for name in ("docling-serve", "docling", "runpod"):
        logger.info("Dependency %s=%s", name, importlib.metadata.version(name))


def error_details(value, payload=None):
    """Keep diagnostics without echoing document bytes or request credentials."""
    sensitive_keys = {"input", "base64_string", "headers", "authorization", "api_key",
                      "x_api_key", "password", "secret", "token", "access_token",
                      "access_key", "secret_key", "secret_access_key"}
    secrets = set()
    api_key = os.getenv("DOCLING_SERVE_API_KEY")
    if api_key:
        secrets.add(api_key)

    def collect(item, sensitive=False):
        if isinstance(item, dict):
            for key, child in item.items():
                collect(child, sensitive or key.lower().replace("-", "_") in sensitive_keys)
        elif isinstance(item, list):
            for child in item:
                collect(child, sensitive)
        elif sensitive and isinstance(item, str) and item:
            secrets.add(item)

    collect(payload)

    def redact(item):
        if isinstance(item, dict):
            return {key: "[redacted]" if key.lower().replace("-", "_") in sensitive_keys
                    else redact(child) for key, child in item.items()}
        if isinstance(item, list):
            return [redact(child) for child in item]
        if isinstance(item, str):
            for secret in sorted(secrets, key=len, reverse=True):
                item = item.replace(secret, "[redacted]")
        return item

    return redact(value)


class ServiceError(RuntimeError):
    def __init__(self, status, path, details):
        super().__init__(f"Docling Serve HTTP {status} at {path}: "
                         + json.dumps(details, ensure_ascii=False))
        self.status = status
        self.path = path
        self.details = details


class DoclingService:
    def __init__(self):
        self.port = int(os.getenv("DOCLING_RUNPOD_PORT", "5001"))
        self.url = f"http://127.0.0.1:{self.port}"
        self.startup_timeout = float(os.getenv("DOCLING_RUNPOD_STARTUP_TIMEOUT", "600"))
        self.job_timeout = float(os.getenv("DOCLING_RUNPOD_JOB_TIMEOUT", "3600"))
        self.poll_interval = float(os.getenv("DOCLING_RUNPOD_POLL_INTERVAL", "2"))
        if min(self.startup_timeout, self.job_timeout, self.poll_interval) <= 0:
            raise ValueError("Worker timeouts and poll interval must be positive")
        self.process = None
        # One conversion at a time: restarting after timeout must not kill another job.
        self.lock = threading.RLock()
        self.opener = build_opener(ProxyHandler({}))

    def request(self, path, payload=None, timeout=30):
        headers = {"Accept": "application/json"}
        if os.getenv("DOCLING_SERVE_API_KEY"):
            headers["X-Api-Key"] = os.environ["DOCLING_SERVE_API_KEY"]
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.url + path, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=timeout) as response:
                body = response.read()
                if response.headers.get_content_type() == "application/zip":
                    return {"filename": "result.zip", "mime_type": "application/zip",
                            "base64_string": base64.b64encode(body).decode("ascii")}
                return json.loads(body)
        except HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
                try:
                    details = json.loads(body)
                except ValueError:
                    details = body or str(exc.reason)
                details = error_details(details, payload)
            finally:
                exc.close()
            raise ServiceError(exc.code, path, details) from None

    def stop(self):
        with self.lock:
            process, self.process = self.process, None
            if process is not None and process.poll() is None:
                logger.info("Stopping Docling Serve pid=%s", process.pid)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    def start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED="1", DOCLING_SERVE_ENABLE_UI="false",
                       DOCLING_SERVE_ENG_KIND="local", UVICORN_ROOT_PATH="",
                       UVICORN_RELOAD="false")
            env.setdefault("DOCLING_SERVE_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))
            # Inherit stdout/stderr so native libraries and the service reach RunPod logs.
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", "docling_serve", "run",
                 "--host", "127.0.0.1", "--port", str(self.port), "--workers", "1"],
                env=env, start_new_session=True,
            )
            logger.info("Starting Docling Serve pid=%s", self.process.pid)
            deadline = time.monotonic() + self.startup_timeout
            try:
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(f"Docling Serve exited with code {self.process.returncode}")
                    try:
                        self.request("/ready", timeout=min(2, max(.1, deadline - time.monotonic())))
                        logger.info("Docling Serve ready")
                        return
                    except (URLError, TimeoutError, ServiceError):
                        time.sleep(1)
                raise TimeoutError("Docling Serve startup timed out")
            except BaseException:
                self.stop()
                raise

    def convert(self, payload, job_id):
        with self.lock:
            self.start()
            deadline = time.monotonic() + self.job_timeout
            def request(path, data=None):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Docling conversion exceeded DOCLING_RUNPOD_JOB_TIMEOUT")
                if self.process.poll() is not None:
                    raise RuntimeError("Docling Serve exited during conversion")
                return self.request(path, data, timeout=min(30, remaining))
            try:
                task = request("/v1/convert/source/async", payload)
                task_id = quote(task["task_id"], safe="")
                previous = None
                last_log = 0
                while True:
                    progress = (task["task_status"], task.get("task_meta"))
                    now = time.monotonic()
                    if progress != previous or now - last_log >= 30:
                        logger.info("job=%s task=%s status=%s progress=%s", job_id,
                                    task_id, progress[0], progress[1])
                        previous, last_log = progress, now
                    if task["task_status"] in _TERMINAL:
                        break
                    time.sleep(min(self.poll_interval, max(0, deadline - now)))
                    task = request(f"/v1/status/poll/{task_id}")
                result = request(f"/v1/result/{task_id}")
                if task["task_status"] in {"failure", "skipped"}:
                    return {"error": "Docling conversion " + task["task_status"],
                            "task_id": task_id, "result": result}
                return result
            except ServiceError:
                raise
            except Exception:
                # No reliable cancellation API: stop timed-out/uncertain work before reuse.
                self.stop()
                raise


def prepare_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Job input must be a Docling Serve conversion request object")
    payload = dict(payload)
    if not isinstance(payload.get("sources"), list) or not payload["sources"]:
        raise ValueError("input.sources must be a non-empty list")
    target = payload.setdefault("target", {"kind": "inbody"})
    if not isinstance(target, dict):
        raise ValueError("input.target must be an object")
    if target.get("kind") == "presigned_url":
        raise ValueError("presigned_url points to worker-local storage; use inbody, zip, or an external storage target")
    return payload


service = DoclingService()
atexit.register(service.stop)


def handler(job):
    job_id = str(job.get("id", "local"))
    started = time.monotonic()
    logger.info("job=%s received", job_id)
    try:
        result = service.convert(prepare_payload(job.get("input")), job_id)
        logger.info("job=%s finished elapsed=%.2fs error=%s", job_id,
                    time.monotonic() - started, "error" in result)
        return result
    except ServiceError as exc:
        logger.warning("job=%s rejected: %s", job_id, exc)
        return {"error": str(exc), "status_code": exc.status,
                "path": exc.path, "details": exc.details, "job_id": job_id}
    except ValueError as exc:
        logger.warning("job=%s rejected: %s", job_id, exc)
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("job=%s conversion failed elapsed=%.2fs", job_id,
                         time.monotonic() - started)
        details = error_details(str(exc), job.get("input"))
        return {"error": f"Docling worker {type(exc).__name__}: {details}", "job_id": job_id}


def main():
    configure_logging()
    verify_ocr()
    service.start()
    import runpod
    try:
        runpod.serverless.start({"handler": handler})
    finally:
        service.stop()


if __name__ == "__main__":
    main()
