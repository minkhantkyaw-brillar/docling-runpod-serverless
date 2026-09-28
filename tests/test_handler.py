"""Offline checks for error propagation from the bundled Docling HTTP service."""

from io import BytesIO
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import handler


class ServiceErrorTests(unittest.TestCase):
    def setUp(self):
        self.service = handler.DoclingService()
        self.service.opener = Mock()

    def fail_request(self, body, status=422, reason="Unprocessable Entity"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        stream = BytesIO(body)
        self.service.opener.open.side_effect = HTTPError(
            "http://127.0.0.1:5001/v1/convert/source/async",
            status, reason, {}, stream,
        )
        return stream

    def test_validation_details_reach_response_and_logs_without_input(self):
        diagnostic = {"detail": [{
            "type": "value_error",
            "loc": ["body", "options"],
            "msg": "Value error, Cannot specify both ocr_preset and ocr_custom_config.",
            "input": {"sources": [{"base64_string": "private-file-bytes"}]},
            "ctx": {"error": "Conflicting OCR configuration"},
        }]}
        stream = self.fail_request(diagnostic)
        job = {"id": "test-422", "input": {"sources": [{"kind": "file"}]}}
        with patch.object(handler, "service", self.service), \
                patch.object(self.service, "convert", side_effect=lambda payload, job_id:
                             self.service.request("/v1/convert/source/async", payload)), \
                self.assertLogs("docling_runpod", level="WARNING") as logs:
            result = handler.handler(job)
        self.assertEqual(result["status_code"], 422)
        self.assertEqual(result["path"], "/v1/convert/source/async")
        self.assertEqual(result["job_id"], "test-422")
        error = result["details"]["detail"][0]
        self.assertEqual(error["loc"], ["body", "options"])
        self.assertEqual(error["msg"], diagnostic["detail"][0]["msg"])
        self.assertEqual(error["ctx"], diagnostic["detail"][0]["ctx"])
        self.assertEqual(error["input"], "[redacted]")
        self.assertIn("Cannot specify both", logs.output[0])
        self.assertNotIn("private-file-bytes", json.dumps(result) + str(logs.output))
        self.assertTrue(stream.closed)

    def test_plaintext_errors_from_each_endpoint(self):
        for path in ("/v1/convert/source/async", "/v1/status/poll/task", "/v1/result/task"):
            with self.subTest(path=path):
                self.fail_request(b"CUDAExecutionProvider initialization failed", 500)
                with self.assertRaises(handler.ServiceError) as caught:
                    self.service.request(path)
                self.assertEqual(caught.exception.status, 500)
                self.assertEqual(caught.exception.path, path)
                self.assertEqual(caught.exception.details,
                                 "CUDAExecutionProvider initialization failed")

    def test_empty_error_body_uses_reason(self):
        self.fail_request(b"", 503, "Service Unavailable")
        with self.assertRaises(handler.ServiceError) as caught:
            self.service.request("/ready")
        self.assertEqual(caught.exception.details, "Service Unavailable")

    def test_malformed_json_and_non_utf8_are_still_reported(self):
        self.fail_request(b'{"detail": "invalid response\xff', 502)
        with self.assertRaises(handler.ServiceError) as caught:
            self.service.request("/v1/result/task")
        self.assertIn("invalid response", caught.exception.details)

    def test_credentials_and_echoed_bytes_are_redacted_in_text(self):
        self.fail_request(b"Rejected private-document-bytes with key service-secret", 500)
        with patch.dict(os.environ, {"DOCLING_SERVE_API_KEY": "service-secret"}), \
                self.assertRaises(handler.ServiceError) as caught:
            self.service.request("/v1/convert/source/async", {
                "sources": [{"base64_string": "private-document-bytes"}],
            })
        self.assertEqual(caught.exception.details, "Rejected [redacted] with key [redacted]")
        nested = handler.error_details({"detail": {"headers": {"Authorization": "secret"},
                                                  "api-key": "secret", "msg": "failed"}})
        self.assertEqual(nested["detail"], {"headers": "[redacted]", "api-key": "[redacted]",
                                            "msg": "failed"})

    def test_worker_exception_is_returned(self):
        with patch.object(handler.service, "convert", side_effect=URLError("connection refused")), \
                self.assertLogs("docling_runpod", level="ERROR"):
            result = handler.handler({"id": "offline", "input": {"sources": [{}]}})
        self.assertIn("URLError", result["error"])
        self.assertIn("connection refused", result["error"])

    def test_successful_response_is_unchanged(self):
        response = Mock()
        response.read.return_value = b'{"task_id":"task","task_status":"pending"}'
        response.headers.get_content_type.return_value = "application/json"
        self.service.opener.open.return_value.__enter__ = Mock(return_value=response)
        self.service.opener.open.return_value.__exit__ = Mock(return_value=False)
        self.assertEqual(self.service.request("/v1/convert/source/async", {"sources": [{}]}),
                         {"task_id": "task", "task_status": "pending"})


if __name__ == "__main__":
    unittest.main()
