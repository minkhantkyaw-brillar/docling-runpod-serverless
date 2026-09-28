#!/usr/bin/env python3
"""Run explicit, resumable Docling experiments; never retry ambiguous submissions.

Plans and results can contain private document data. Keep them under tests/results.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

from test_endpoint import Client, Case, endpoint_url, source_for, validate, values, ROOT

BACKENDS = ['docling_parse', '_docling_parse', 'dlparse_v4', 'dlparse_v2', 'dlparse_v1', 'pypdfium2']
ENGINES = ['tesseract', 'easyocr', 'rapidocr']
BASE = {'to_formats': ['md', 'json'], 'pipeline': 'standard',
        'do_table_structure': True, 'table_mode': 'accurate', 'table_cell_matching': True,
        'include_images': False, 'image_export_mode': 'placeholder',
        'do_code_enrichment': False, 'do_formula_enrichment': False,
        'do_picture_classification': False, 'do_picture_description': False}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def tokens(text):
    return re.findall(r'\w+(?:[.,/-]\w+)*', text.casefold())


def token_score(reference, candidate):
    """Order independent diagnostic, not a measure of table/reading-order accuracy."""
    a, b = Counter(tokens(reference)), Counter(tokens(candidate))
    common = sum((a & b).values())
    precision = common / sum(b.values()) if b else 0
    recall = common / sum(a.values()) if a else 0
    return {'token_precision': precision, 'token_recall': recall,
            'token_f1': 2 * precision * recall / (precision + recall) if precision + recall else 0}


def document_text(doc):
    parts = [x.get('text', '') for x in doc.get('texts', [])]
    for table in doc.get('tables', []):
        parts.extend(c.get('text', '') for c in table.get('data', {}).get('table_cells', []))
    return '\n'.join(parts)


def without_bboxes(value):
    """Keep text, order, page provenance and table cells; ignore only coordinate boxes."""
    if isinstance(value, dict):
        return {k: without_bboxes(v) for k, v in value.items() if k != 'bbox'}
    if isinstance(value, list):
        return [without_bboxes(v) for v in value]
    return value


def analyze(output, references):
    docs = list(values(output, 'json_content'))
    if not docs or not all(isinstance(d, dict) for d in docs):
        raise ValueError('Missing structured documents')
    results = []
    for doc in docs:
        name = doc.get('origin', {}).get('filename', doc.get('name', ''))
        ref = references.get(name, references.get(Path(name).stem, {}))
        if ref.get('segments'):
            for segment in ref['segments']:
                pages = set(segment['pages'])
                part = dict(doc, origin={'filename': segment['filename']})
                part['pages'] = {k: v for k, v in doc.get('pages', {}).items() if int(k) in pages}
                for field in ('texts', 'tables', 'pictures'):
                    part[field] = [item for item in doc.get(field, [])
                                   if any(p.get('page_no') in pages for p in item.get('prov', []))]
                # Whole-document structural stability is recorded separately by the caller.
                part.pop('body', None)
                part.pop('furniture', None)
                part.pop('groups', None)
                results.extend(analyze({'json_content': part}, {segment['filename']: segment}))
            continue
        text = document_text(doc)
        normalized = ' '.join(text.casefold().split())
        anchors = ref.get('anchors', [])
        found = [a for a in anchors if ' '.join(a.casefold().split()) in normalized]
        clean = {k: v for k, v in doc.items() if k not in ('origin', 'name')}
        row = {'filename': name, 'json_hash': digest(clean), 'text_hash': digest(text),
               'semantic_hash': digest(without_bboxes(clean)),
               'text_characters': len(text), 'pages': len(doc.get('pages', {})),
               'tables': len(doc.get('tables', [])), 'anchors_found': len(found),
               'anchors_total': len(anchors), 'missing_anchors': [a for a in anchors if a not in found]}
        if ref.get('native_text'):
            row.update(token_score(ref['native_text'], text))
        results.append(row)
    return results


def summarize(folder):
    report = json.loads((folder / 'report.json').read_text())
    groups = defaultdict(list)
    for row in report['runs']:
        groups[row['config']].append(row)
    summary = []
    for config, runs in groups.items():
        by_doc = defaultdict(list)
        for run in runs:
            for doc in run.get('documents', []):
                by_doc[doc['filename']].append(doc)
        summary.append({'config': config, 'runs': len(runs),
                        'passed': sum(r['status'] == 'PASS' for r in runs),
                        'execution_seconds': [r.get('execution_ms', 0) / 1000 for r in runs],
                        'documents': [{
                            'filename': name, 'observations': len(docs),
                            'unique_json': len({d['json_hash'] for d in docs}),
                            'unique_text': len({d['text_hash'] for d in docs}),
                            'token_f1': [round(d['token_f1'], 5) for d in docs if 'token_f1' in d],
                            'anchors_found': [d['anchors_found'] for d in docs],
                            'anchors_total': docs[0]['anchors_total'],
                            'missing_anchors': [d['missing_anchors'] for d in docs],
                            'tables': [d['tables'] for d in docs],
                        } for name, docs in sorted(by_doc.items())]})
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    for s in summary:
        print(s['config'], f"{s['passed']}/{s['runs']} passed",
              ' | '.join(f"{d['filename']}: JSON {d['unique_json']}/{d['observations']} "
                         f"F1={d['token_f1']} anchors={d['anchors_found']}/{d['anchors_total']}"
                         for d in s['documents']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--inspect-batch-api', action='store_true')
    parser.add_argument('--batch', action='store_true', help='Use RunPod Batch API, one file per child request')
    parser.add_argument('--cancel-batch-report', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output_dir)
        return 0
    if args.repeats < 1 or args.timeout <= 0:
        parser.error('Repeats and timeout must be positive')
    plan = json.loads(args.plan.read_text())
    tasks = []
    for rep in range(args.repeats):
        configs = list(plan['configs'])
        random.Random(927 + rep).shuffle(configs)
        tasks.extend((rep, config) for config in configs)
    if args.list:
        count = sum(len(c.get('sources', plan['sources'])) for _, c in tasks) if args.batch else len(tasks)
        print(json.dumps({'jobs': count, 'configs': plan['configs']}, indent=2))
        return 0
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    key = os.getenv('RUNPOD_API_KEY', '').strip()
    if not key:
        parser.error('RUNPOD_API_KEY is missing')
    client = Client(endpoint_url(os.getenv('RUNPOD_ENDPOINT_URL', '')), key, args.timeout, 3)
    if args.cancel_batch_report:
        for path in args.cancel_batch_report:
            report = json.loads(path.read_text())
            result = client.request('POST', '/batch/' + report['batch_id'] + '/cancel')
            report['cancellation_response'] = result
            path.write_text(json.dumps(report, indent=2))
            print('Cancelled batch', report['batch_id'])
        return 0
    if args.inspect_batch_api:
        for path in ('/health', '/batch'):
            try:
                response = client.request('GET', path)
                print(path, json.dumps(response)[:2000])
            except RuntimeError as exc:
                print(path, str(exc))
        return 0
    if args.batch:
        from runpod_batch import run_batch
        return run_batch(client, plan, args.output_dir, args.repeats, args.timeout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / 'report.json'
    fingerprint = digest(plan)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['plan_hash'] != fingerprint:
            raise ValueError('Plan changed; use another output directory')
        if any(r['status'] in {'RUNNING', 'UNCERTAIN', 'INTERRUPTED'} for r in report['runs']):
            raise ValueError('Unresolved submission in report; inspect the saved job before continuing')
    else:
        report = {'plan_hash': fingerprint, 'created': datetime.now(timezone.utc).isoformat(),
                  'plan': plan, 'runs': []}
    def save():
        temp = report_path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(report_path)
    done = {r['id'] for r in report['runs']}
    try:
        for rep, config in tasks:
            run_id = f"{config['name']}-r{rep + 1}"
            if run_id in done:
                continue
            sources = [Path(p) for p in config.get('sources', plan['sources'])]
            options = dict(BASE, **config['options'])
            payload = {'sources': [source_for(Case('', p)) for p in sources],
                       'options': options, 'target': {'kind': 'inbody'}}
            row = {'id': run_id, 'config': config['name'], 'repeat': rep + 1,
                   'options': options, 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                   'status': 'RUNNING', 'started': datetime.now(timezone.utc).isoformat()}
            report['runs'].append(row)
            save()
            print('RUN', run_id, flush=True)
            started = time.monotonic()
            response = None
            def submitted(job_id):
                row['job_id'] = job_id
                save()
            try:
                response = client.run(payload, submitted)
                (args.output_dir / f'{run_id}.json').write_text(json.dumps(response, indent=2))
                row.update(execution_ms=response.get('executionTime', 0),
                           delay_ms=response.get('delayTime', 0), worker_id=response.get('workerId'))
                validate(Case('', sources[0], options, check='exports'), response)
                row['documents'] = analyze(response['output'], plan.get('references', {}))
                if len(list(values(response['output'], 'json_content'))) != len(sources):
                    raise ValueError('Returned document count differs from source count')
                if not all(d['text_characters'] for d in row['documents']) and options.get('do_ocr'):
                    raise ValueError('OCR-enabled conversion returned empty document text')
                row['markdown_hash'] = digest(list(values(response['output'], 'md_content')))
                row['document_hash'] = digest(list(values(response['output'], 'json_content')))
                row['status'] = 'PASS'
            except Exception as exc:
                row.update(status='FAIL' if response is not None else 'UNCERTAIN', reason=str(exc))
                if client.active_job:
                    client.cancel()
                print(row['status'], str(exc), flush=True)
            row['elapsed_seconds'] = round(time.monotonic() - started, 3)
            save()
            print(row['status'], run_id, row['elapsed_seconds'], flush=True)
            if row['status'] == 'UNCERTAIN':
                break
            if len(report['runs']) >= 3 and all(r['status'] == 'FAIL' for r in report['runs'][-3:]):
                print('Stopping after three consecutive failures; inspect responses before resuming.')
                break
    except KeyboardInterrupt:
        client.cancel()
        row['status'] = 'INTERRUPTED'
        save()
        return 130
    summarize(args.output_dir)
    return int(any(r['status'] != 'PASS' for r in report['runs']))


if __name__ == '__main__':
    sys.exit(main())
