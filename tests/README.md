# RunPod endpoint tests

Install the test dependencies, then put these entries in the repository `.env`:

```dotenv
RUNPOD_ENDPOINT_URL=
RUNPOD_API_KEY=
TEST_SOURCE_DIR=
```

The blank template is also in [endpoint.env.example](../endpoint.env.example). Set the URL to `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID` (a trailing `/run` or `/runsync` is accepted). Existing environment variables take precedence over `.env`. `TEST_SOURCE_DIR` is required. It accepts an absolute path or a path relative to the repository root (quote paths containing spaces). Prefer a source directory outside this repository. `--list` loads configuration and lists filenames, but does not require endpoint credentials, read document contents, or contact RunPod. Offline unit tests do not load `.env`.

```bash
python3 -m pip install -r tests/requirements.txt
# Preview the matrix without reading document contents or contacting RunPod:
python3 tests/test_endpoint.py --list
# Once the endpoint is ready, start with one job:
python3 tests/test_endpoint.py --smoke
# Run the standard cases for your configured documents sequentially:
python3 tests/test_endpoint.py
```

`--smoke` submits exactly one job: the smallest PDF with only `to_formats: ["md"]`.

Source documents come only from `TEST_SOURCE_DIR`; there is no built-in source folder or bundled dataset. The runner discovers PDF and XLSX files directly in that directory (not recursively), and requires at least one PDF. The number of cases depends on your files. Live tests send the selected source bytes to your configured endpoint.

Local data directories under `tests/` and generated results are excluded from Git; only the test scripts, README, and requirements file are allowed there. Keep other source directories outside the repository, or add their location to `.gitignore` before use. The `.env` file is also excluded. Source documents are not required in Git.

## What is checked

- Each discovered original: Markdown/JSON output and nonempty document text.
- Each PDF's first page rendered to a PNG at 150 DPI, tested with **Tesseract, RapidOCR, EasyOCR, and Tesseract CLI**. `force_ocr=true`, explicit engine selection, and English language configuration exercise each backend on image-only content. At least 20 alphanumeric output characters are required; filenames and image placeholders do not count.
- OCR disabled on the same raster fixture: must produce no document text.
- Page ranges: returned JSON page numbers must match page 1 or page 2 exactly.
- Fast tables with cell matching off, tables disabled, alternative PDF backend, and OCR disabled on a native PDF.
- Markdown, JSON, HTML, text, YAML, and DocTags export fields.
- Embedded page-image data and referenced image assets in a valid base64 ZIP archive.
- Invalid OCR engine, format, and page range: must produce explicit validation errors. Generic crashes and timeouts do not count as correct rejection.
- Another valid job after invalid requests to check worker reuse.

These are functional smoke/behavior checks, not ground-truth OCR accuracy scores or proof of GPU acceleration. The API does not report the selected backend's identity, so worker logs should corroborate explicit engine selection. Table-mode cases check successful conversion; they do not measure table accuracy. The page-two test requires at least two pages in the first PDF. Full baseline jobs process complete originals; most option checks use one page.

Optional model-heavy cases exercise code/formula enrichment, picture classification/description, chart extraction, heading hierarchy, and the Granite Docling VLM pipeline:

```bash
python3 tests/test_endpoint.py --extended --list
python3 tests/test_endpoint.py --extended
```

Extended cases check that conversion succeeds with the feature enabled. They do not prove feature quality when the selected page lacks the corresponding content. Missing models, insufficient GPU memory, and service policy restrictions fail rather than silently skip these cases. This matrix does **not** claim to exhaust all 51 options, nested configurations, external storage, callbacks, or option combinations.

## Results and failures

The worker image enables `DOCLING_SERVE_ALLOW_CUSTOM_OCR_CONFIG=true`,
`DOCLING_SERVE_DEBUG_ERROR_DETAILS=true`, and `DOCLING_DEVICE=cuda` by default.
Rebuild the image and roll out new workers to apply these defaults; live workers
keep their existing image and environment. Explicit deployment environment values
override image defaults. For local CPU smoke tests, set `DOCLING_DEVICE=cpu`.

Docling HTTP failures return `error`, `status_code`, `path`, `details`, and `job_id`
in the RunPod job output. `details` preserves the upstream JSON error body (including
422 field locations and validation messages), or the response text for non-JSON
errors. The same details are logged with the job ID. Echoed validation inputs,
file bytes, and credential fields are redacted. Worker exceptions also return
their exception type and message. Detailed errors may include service tracebacks.

Results go to `tests/results/<UTC timestamp>/`: a `report.json` with case names, inputs, options, RunPod job IDs, elapsed times and outcomes, plus each completed job's JSON response. The report is saved immediately after submission so a job can be found if the test runner is interrupted. Response files contain extracted source-document data; this folder is git-ignored. The API key and full base64 input are not written to the report.

Exit codes: `0` all checks passed, `1` test failures, `2` configuration errors, `130` interruption. `--match ocr-tesseract-` selects the Python Tesseract cases; `--match ocr-rapidocr-` and `--match ocr-easyocr-` select the other backends. Use `--timeout 3600` to extend the client wait per job (includes queue/cold-start time); this does not change the endpoint's execution-time limit. `--poll-interval` defaults to five seconds.

The runner follows RunPod's [async request protocol](https://docs.runpod.io/serverless/endpoints/send-requests): submit once to `/run`, then poll `/status/{id}`. It never automatically resubmits ambiguous POST requests. On polling failures, timeout, or Ctrl-C it attempts cancellation of only its active job. If cancellation cannot be confirmed, it stops further submissions and prints the job ID for manual inspection. A failed submission without a returned ID must be checked in RunPod before rerunning. Live invocations consume endpoint compute.

Offline verification (does not read `.env` or send jobs):

```bash
python3 -m unittest discover -s tests -p test_endpoint_runner.py -v
```

## Accuracy and repeatability benchmarks

`benchmark_backends.py` accepts a JSON experiment plan. Each configuration has a
`name` and Docling `options`, and may override the plan's `sources` list. References
may contain manually verified `anchors` or a `native_text` transcript, keyed by
source filename. Keep plans and references under the ignored `tests/results/`
directory because they may contain private document data.

```json
{
  "sources": ["/absolute/path/to/document.pdf"],
  "configs": [
    {
      "name": "scan-rapidocr",
      "options": {
        "pdf_backend": "pypdfium2",
        "ocr_preset": "rapidocr",
        "ocr_lang": ["en"],
        "do_ocr": true,
        "force_ocr": true
      }
    }
  ]
}
```

```bash
# Preview without submitting jobs:
python3 tests/benchmark_backends.py tests/results/plan.json \
  --output-dir tests/results/comparison --repeats 3 --list
# Dedicated RunPod Batch API: EACH child job contains exactly ONE source file.
python3 tests/benchmark_backends.py tests/results/plan.json \
  --output-dir tests/results/comparison --repeats 3 --batch --timeout 3600
# Re-score saved responses without API access, including structural comparisons:
python3 tests/analyze_benchmark.py tests/results/comparison \
  --output tests/results/comparison/analysis.json
# Offline unit checks:
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Batch jobs are created as drafts and finalized only after all child IDs have been
saved. The live API accepts an **array** when appending requests. The runner maps
new child IDs by set difference, rather than assuming API list order matches
submission order. Each configuration/source combination runs the requested number
of times in a reproducible shuffled schedule. Failed/ambiguous POSTs are never
retried automatically. Finalized batches can be monitored again using the same
command and output directory without duplicate submissions. An uncertain draft
mutation requires inspection first. A client timeout or interruption after
finalization leaves the batch running; its ID is saved in `report.json`.
If every child result is terminal and saved, the runner finishes even when the
Batch API's aggregate completion counter lags behind those individual results.

`--list` counts individual child jobs when `--batch` is selected. Omitting `--batch`
uses sequential `/run` calls, for single-source plans. Live execution consumes
endpoint compute. Results include extracted document content; saved batch responses
omit echoed input bytes. The client wait limit does not change worker timeouts.

The analysis distinguishes exact document JSON, exact Markdown, extracted text,
and structured content with only `bbox` coordinate fields removed. The latter
still includes text, reading order, page provenance, table dimensions, cell spans,
and row/column associations. Tiny coordinate differences should not be described
as changed text. Three matching observations are evidence for this deployment and
these files, not a determinism guarantee across hardware, restarts or upgrades.

Native-text token F1 measures order-independent agreement with the PDF text layer;
it is **not** ground-truth OCR accuracy or a table correctness score. Manual anchor
recall measures only the listed values and likewise does not prove correct row or
column assignment. Inspect the source renderings and output tables as well.

Use `ocr_preset` for explicit engine selection. In this endpoint, Python Tesseract
uses `ocr_preset: "tesserocr"`, CLI Tesseract uses `"tesseract"`, and both require
English `ocr_lang: ["eng"]`; EasyOCR and RapidOCR use `["en"]`. The functional
runner's existing `ocr-tesseract-*` and `ocr-tesseract_cli-*` case names remain.
The older `ocr_engine` field is deprecated and its handling varies by version.

The September 2026 local investigation, source-specific plans, responses, rendered
samples and recommendations are in `tests/results/benchmark/`. Those private
artifacts are intentionally excluded from Git. No deployment settings are changed
by the benchmark.
