import base64
import json
import logging
import re
import time
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlparse
from zipfile import ZIP_DEFLATED, ZipFile

import runpod
import yaml
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import DocumentStream, InputFormat, OutputFormat
from docling.datamodel.pipeline_options import (
	PdfBackend,
	PdfPipelineOptions,
	TableFormerMode,
	TableStructureOptions,
	TesseractCliOcrOptions,
	TesseractOcrOptions,
	normalize_pdf_backend,
)
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling.models.factories import get_ocr_factory
from docling_core.types.doc import ImageRefMode


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_FORMAT_ALIASES = {
	"markdown": "md",
	"txt": "text",
	"htm": "html",
}
_ALLOWED_OUTPUT_FORMATS = {fmt.value for fmt in OutputFormat}
_NO_IMAGE_EXPORT_FORMATS = {
	OutputFormat.TEXT.value,
	OutputFormat.DOCTAGS.value,
	OutputFormat.VTT.value,
}


def _as_bool(value: Any, default: bool) -> bool:
	if value is None:
		return default
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)


def _as_float(value: Any, default: float | None = None) -> float | None:
	if value is None:
		return default
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


def _as_int(value: Any, default: int | None = None) -> int | None:
	if value is None:
		return default
	try:
		return int(value)
	except (TypeError, ValueError):
		return default


def _as_list_of_str(value: Any) -> list[str] | None:
	if value is None:
		return None
	if isinstance(value, str):
		items = [part.strip() for part in re.split(r"[;,]", value) if part.strip()]
		return items or None
	if isinstance(value, list):
		items = [str(item).strip() for item in value if str(item).strip()]
		return items or None
	return None


def _normalize_output_formats(value: Any) -> list[str]:
	raw_formats = _as_list_of_str(value) or [OutputFormat.MARKDOWN.value]
	normalized: list[str] = []

	for fmt in raw_formats:
		key = _FORMAT_ALIASES.get(fmt.lower(), fmt.lower())
		if key in _ALLOWED_OUTPUT_FORMATS and key not in normalized:
			normalized.append(key)

	if not normalized:
		normalized = [OutputFormat.MARKDOWN.value]

	return normalized


def _first_non_empty(values: list[Any]) -> str | None:
	for value in values:
		if value is None:
			continue
		text = str(value).strip()
		if text:
			return text
	return None


def _as_path_or_none(value: Any) -> Path | None:
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	return Path(text).expanduser()


def _resolve_artifacts_path(options: dict[str, Any]) -> Path | None:
	explicit_artifacts = _as_path_or_none(
		_first_non_empty([options.get("artifacts_path"), options.get("model_artifacts_path")])
	)
	if explicit_artifacts is None:
		return None
	if not explicit_artifacts.is_dir():
		raise ValueError(
			f"Invalid artifacts path '{explicit_artifacts}'. It must be an existing directory."
		)
	return explicit_artifacts


def _normalize_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
	if isinstance(payload.get("sources"), list) and payload["sources"]:
		return payload["sources"]

	sources: list[dict[str, Any]] = []

	for item in payload.get("http_sources", []) or []:
		source = dict(item)
		source.setdefault("kind", "http")
		sources.append(source)

	for item in payload.get("file_sources", []) or []:
		source = dict(item)
		source.setdefault("kind", "file")
		sources.append(source)

	return sources


def _normalize_target_kind(payload: dict[str, Any]) -> str:
	target = payload.get("target", {"kind": "inbody"})
	if isinstance(target, str):
		return target.lower()
	if isinstance(target, dict):
		return str(target.get("kind", "inbody")).lower()
	return "inbody"


def _normalize_image_mode(raw: Any) -> ImageRefMode:
	mode = str(raw or ImageRefMode.PLACEHOLDER.value).lower()
	if mode not in {m.value for m in ImageRefMode}:
		mode = ImageRefMode.PLACEHOLDER.value
	return ImageRefMode(mode)


def _sanitize_name(name: str, fallback: str) -> str:
	safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
	return safe or fallback


def _source_name(source: dict[str, Any], index: int) -> str:
	kind = str(source.get("kind", "")).lower()

	if kind == "file":
		filename = source.get("filename") or f"source_{index}"
		stem = Path(str(filename)).stem
		return _sanitize_name(stem, f"source_{index}")

	if kind == "http":
		raw_url = str(source.get("url", ""))
		parsed = urlparse(raw_url)
		candidate = Path(parsed.path).stem or parsed.netloc or f"source_{index}"
		return _sanitize_name(candidate, f"source_{index}")

	return _sanitize_name(f"source_{index}", f"source_{index}")


def _prepare_source(source: dict[str, Any], index: int) -> tuple[Any, dict[str, str] | None, str]:
	kind = str(source.get("kind", "")).lower()
	name = _source_name(source, index)

	if kind == "http":
		url = source.get("url")
		if not url:
			raise ValueError("HTTP source requires a non-empty 'url'.")
		headers = source.get("headers")
		if headers is not None and not isinstance(headers, dict):
			raise ValueError("Source headers must be a dictionary.")
		return str(url), headers, name

	if kind == "file":
		b64 = source.get("base64_string")
		if not b64:
			raise ValueError("File source requires a non-empty 'base64_string'.")

		b64_text = str(b64)
		if b64_text.startswith("data:") and "," in b64_text:
			b64_text = b64_text.split(",", 1)[1]

		decoded = base64.b64decode(b64_text)
		filename = source.get("filename") or f"{name}.bin"
		stream = DocumentStream(name=str(filename), stream=BytesIO(decoded))
		return stream, None, name

	raise ValueError(f"Unsupported source kind: {kind!r}")


def _jsonable(value: Any) -> Any:
	if hasattr(value, "model_dump"):
		return value.model_dump(mode="json")
	if isinstance(value, dict):
		return {str(k): _jsonable(v) for k, v in value.items()}
	if isinstance(value, list):
		return [_jsonable(v) for v in value]
	if isinstance(value, tuple):
		return [_jsonable(v) for v in value]
	if isinstance(value, (str, int, float, bool)) or value is None:
		return value
	return str(value)


def _build_converter(options: dict[str, Any], to_formats: list[str]) -> DocumentConverter:
	pipeline_options = PdfPipelineOptions(
		do_ocr=_as_bool(options.get("ocr", options.get("do_ocr")), True),
		do_table_structure=_as_bool(options.get("do_table_structure"), True),
		do_code_enrichment=_as_bool(options.get("do_code_enrichment"), False),
		do_formula_enrichment=_as_bool(options.get("do_formula_enrichment"), False),
		do_picture_classification=_as_bool(options.get("do_picture_classification"), False),
		do_picture_description=_as_bool(options.get("do_picture_description"), False),
		enable_remote_services=_as_bool(options.get("enable_remote_services"), False),
	)

	artifacts_path = _resolve_artifacts_path(options)
	if artifacts_path is not None:
		pipeline_options.artifacts_path = artifacts_path

	timeout = _as_float(options.get("document_timeout"), None)
	if timeout is not None:
		pipeline_options.document_timeout = timeout

	table_mode_raw = str(options.get("table_mode", TableFormerMode.ACCURATE.value)).lower()
	table_mode = TableFormerMode.FAST if table_mode_raw == TableFormerMode.FAST.value else TableFormerMode.ACCURATE

	if isinstance(pipeline_options.table_structure_options, TableStructureOptions):
		pipeline_options.table_structure_options.mode = table_mode
		pipeline_options.table_structure_options.do_cell_matching = _as_bool(
			options.get("table_cell_matching"),
			True,
		)

	ocr_engine = str(options.get("ocr_engine", "easyocr")).strip().lower() or "easyocr"
	ocr_factory = get_ocr_factory(allow_external_plugins=_as_bool(options.get("allow_external_plugins"), False))
	try:
		ocr_options = ocr_factory.create_options(
			kind=ocr_engine,
			force_full_page_ocr=_as_bool(options.get("force_ocr"), False),
		)
	except RuntimeError:
		logger.warning("Unsupported OCR engine '%s', falling back to 'auto'.", ocr_engine)
		ocr_options = ocr_factory.create_options(
			kind="auto",
			force_full_page_ocr=_as_bool(options.get("force_ocr"), False),
		)

	ocr_lang = _as_list_of_str(options.get("ocr_lang"))
	if ocr_lang is not None:
		ocr_options.lang = ocr_lang

	psm = _as_int(options.get("psm"), None)
	if psm is not None and isinstance(ocr_options, (TesseractOcrOptions, TesseractCliOcrOptions)):
		ocr_options.psm = psm

	pipeline_options.ocr_options = ocr_options

	image_mode = _normalize_image_mode(options.get("image_export_mode"))
	if image_mode != ImageRefMode.PLACEHOLDER and any(
		fmt not in _NO_IMAGE_EXPORT_FORMATS for fmt in to_formats
	):
		pipeline_options.generate_page_images = True
		pipeline_options.generate_picture_images = True
		pipeline_options.images_scale = _as_float(options.get("images_scale"), 2.0) or 2.0

	backend_raw = str(options.get("pdf_backend", PdfBackend.DOCLING_PARSE.value)).lower()
	backend: PdfBackend
	try:
		backend = normalize_pdf_backend(PdfBackend(backend_raw))
	except ValueError:
		logger.warning("Unsupported PDF backend '%s', falling back to docling_parse.", backend_raw)
		backend = PdfBackend.DOCLING_PARSE

	backend_cls = (
		PyPdfiumDocumentBackend
		if backend == PdfBackend.PYPDFIUM2
		else DoclingParseDocumentBackend
	)

	return DocumentConverter(
		format_options={
			InputFormat.PDF: PdfFormatOption(
				pipeline_options=pipeline_options,
				backend=backend_cls,
			),
			InputFormat.IMAGE: ImageFormatOption(
				pipeline_options=pipeline_options,
			),
		}
	)


def _save_html_to_string(document: Any, image_mode: ImageRefMode, split_page_view: bool) -> str:
	with NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as tmp:
		path = Path(tmp.name)

	try:
		document.save_as_html(
			filename=path,
			image_mode=image_mode,
			split_page_view=split_page_view,
		)
		return path.read_text(encoding="utf-8")
	finally:
		path.unlink(missing_ok=True)


def _save_vtt_to_string(document: Any) -> str:
	with NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8") as tmp:
		path = Path(tmp.name)

	try:
		document.save_as_vtt(filename=path)
		return path.read_text(encoding="utf-8")
	finally:
		path.unlink(missing_ok=True)


def _export_document(document: Any, to_formats: list[str], image_mode: ImageRefMode) -> dict[str, Any]:
	content: dict[str, Any] = {
		"md_content": None,
		"json_content": None,
		"yaml_content": None,
		"html_content": None,
		"html_split_page_content": None,
		"text_content": None,
		"doctags_content": None,
		"vtt_content": None,
	}

	if OutputFormat.MARKDOWN.value in to_formats:
		try:
			content["md_content"] = document.export_to_markdown(image_mode=image_mode)
		except TypeError:
			content["md_content"] = document.export_to_markdown()

	if OutputFormat.JSON.value in to_formats:
		content["json_content"] = _jsonable(document.export_to_dict())

	if OutputFormat.YAML.value in to_formats:
		content["yaml_content"] = yaml.safe_dump(
			document.export_to_dict(),
			sort_keys=False,
			allow_unicode=False,
		)

	if OutputFormat.HTML.value in to_formats:
		try:
			content["html_content"] = document.export_to_html(image_mode=image_mode)
		except TypeError:
			content["html_content"] = _save_html_to_string(
				document=document,
				image_mode=image_mode,
				split_page_view=False,
			)

	if OutputFormat.HTML_SPLIT_PAGE.value in to_formats:
		content["html_split_page_content"] = _save_html_to_string(
			document=document,
			image_mode=image_mode,
			split_page_view=True,
		)

	if OutputFormat.TEXT.value in to_formats:
		content["text_content"] = document.export_to_markdown(strict_text=True)

	if OutputFormat.DOCTAGS.value in to_formats:
		content["doctags_content"] = document.export_to_doctags()

	if OutputFormat.VTT.value in to_formats:
		content["vtt_content"] = _save_vtt_to_string(document=document)

	return content


def _source_failure(name: str, message: str) -> dict[str, Any]:
	return {
		"name": name,
		"document": {
			"md_content": None,
			"json_content": None,
			"yaml_content": None,
			"html_content": None,
			"html_split_page_content": None,
			"text_content": None,
			"doctags_content": None,
			"vtt_content": None,
		},
		"status": "failure",
		"processing_time": 0.0,
		"timings": {},
		"errors": [{"error_message": message}],
	}


def _status_summary(results: list[dict[str, Any]]) -> str:
	if not results:
		return "failure"

	statuses = {str(item.get("status", "")).lower() for item in results}
	if statuses == {"success"}:
		return "success"
	if statuses <= {"failure"}:
		return "failure"
	if "failure" in statuses:
		return "partial_success"
	return next(iter(statuses))


def _write_zip_payload(results: list[dict[str, Any]]) -> str:
	buffer = BytesIO()
	with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as zf:
		manifest = []

		for item in results:
			base = item["name"]
			document = item.get("document", {})
			status = item.get("status", "unknown")

			if status != "failure":
				if document.get("md_content") is not None:
					zf.writestr(f"{base}.md", document["md_content"])
				if document.get("json_content") is not None:
					zf.writestr(
						f"{base}.json",
						json.dumps(document["json_content"], ensure_ascii=True, indent=2),
					)
				if document.get("yaml_content") is not None:
					zf.writestr(f"{base}.yaml", document["yaml_content"])
				if document.get("html_content") is not None:
					zf.writestr(f"{base}.html", document["html_content"])
				if document.get("html_split_page_content") is not None:
					zf.writestr(
						f"{base}.split_page.html",
						document["html_split_page_content"],
					)
				if document.get("text_content") is not None:
					zf.writestr(f"{base}.txt", document["text_content"])
				if document.get("doctags_content") is not None:
					zf.writestr(f"{base}.doctags", document["doctags_content"])
				if document.get("vtt_content") is not None:
					zf.writestr(f"{base}.vtt", document["vtt_content"])

			manifest.append(
				{
					"name": base,
					"status": status,
					"processing_time": item.get("processing_time", 0.0),
					"errors": item.get("errors", []),
				}
			)

		zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=True, indent=2))

	return base64.b64encode(buffer.getvalue()).decode("ascii")


def process_request(payload: dict[str, Any]) -> dict[str, Any]:
	started = time.time()

	sources = _normalize_sources(payload)
	if not sources:
		raise ValueError("No input sources provided. Use 'sources' with 'http' or 'file' items.")

	options = payload.get("options") or {}
	if not isinstance(options, dict):
		raise ValueError("'options' must be an object.")

	to_formats = _normalize_output_formats(options.get("to_formats"))
	image_mode = _normalize_image_mode(options.get("image_export_mode"))
	target_kind = _normalize_target_kind(payload)
	abort_on_error = _as_bool(options.get("abort_on_error"), False)

	converter = _build_converter(options=options, to_formats=to_formats)

	page_range = options.get("page_range")
	convert_kwargs: dict[str, Any] = {"raises_on_error": False}
	if isinstance(page_range, list) and len(page_range) == 2:
		start_page = _as_int(page_range[0], None)
		end_page = _as_int(page_range[1], None)
		if start_page is not None and end_page is not None:
			convert_kwargs["page_range"] = (start_page, end_page)

	max_num_pages = _as_int(options.get("max_num_pages"), None)
	if max_num_pages is not None:
		convert_kwargs["max_num_pages"] = max_num_pages

	max_file_size = _as_int(options.get("max_file_size"), None)
	if max_file_size is not None:
		convert_kwargs["max_file_size"] = max_file_size

	results: list[dict[str, Any]] = []

	for index, source in enumerate(sources, start=1):
		name = _source_name(source, index)
		source_start = time.time()

		try:
			source_input, headers, name = _prepare_source(source, index)

			result = converter.convert(
				source=source_input,
				headers=headers,
				**convert_kwargs,
			)

			status = str(getattr(result.status, "value", result.status)).lower()
			source_response = {
				"name": name,
				"document": {
					"md_content": None,
					"json_content": None,
					"yaml_content": None,
					"html_content": None,
					"html_split_page_content": None,
					"text_content": None,
					"doctags_content": None,
					"vtt_content": None,
				},
				"status": status,
				"processing_time": round(time.time() - source_start, 6),
				"timings": _jsonable(getattr(result, "timings", {})),
				"errors": _jsonable(getattr(result, "errors", [])),
			}

			if getattr(result, "document", None) is not None:
				source_response["document"] = _export_document(
					document=result.document,
					to_formats=to_formats,
					image_mode=image_mode,
				)

			results.append(source_response)

			if abort_on_error and status == "failure":
				break

		except Exception as exc:
			logger.exception("Failed processing source %s", name)
			fail_response = _source_failure(name=name, message=str(exc))
			fail_response["processing_time"] = round(time.time() - source_start, 6)
			results.append(fail_response)
			if abort_on_error:
				break

	processing_time = round(time.time() - started, 6)
	overall_status = _status_summary(results)

	# Mimic /v1/convert/source in-body response whenever we have exactly one source
	# and the target is not zip.
	if len(results) == 1 and target_kind != "zip":
		only = results[0]
		return {
			"document": only["document"],
			"status": only["status"],
			"processing_time": processing_time,
			"timings": only.get("timings", {}),
			"errors": only.get("errors", []),
		}

	zip_b64 = _write_zip_payload(results)

	failed = [item for item in results if item.get("status") == "failure"]
	succeeded = [item for item in results if item.get("status") != "failure"]

	return {
		"status": overall_status,
		"processing_time": processing_time,
		"num_converted": len(results),
		"num_succeeded": len(succeeded),
		"num_failed": len(failed),
		"result": {
			"kind": "zip",
			"content_type": "application/zip",
			"filename": "converted_docs.zip",
			"zip_base64": zip_b64,
		},
		"errors": [
			{
				"source": item.get("name"),
				"errors": item.get("errors", []),
			}
			for item in failed
		],
	}


def handler(job: dict[str, Any]) -> dict[str, Any]:
	payload = job.get("input") or {}
	if not isinstance(payload, dict):
		return {"error": "Job input must be a JSON object."}

	# Allows clients to send {"payload": {...}} while still supporting direct bodies.
	if "sources" not in payload and "payload" in payload and isinstance(payload["payload"], dict):
		payload = payload["payload"]

	try:
		return process_request(payload)
	except Exception as exc:
		logger.exception("Job failed")
		return {"error": str(exc)}


runpod.serverless.start({"handler": handler})