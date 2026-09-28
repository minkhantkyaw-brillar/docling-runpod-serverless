"""Dedicated RunPod Batch API driver. Every child request contains exactly one file."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import quote

from test_endpoint import Case, source_for, validate, values


def children(client, batch_id):
    rows, offset = [], 0
    while True:
        result = client.request('GET', f'/batch/{quote(batch_id, safe="")}/requests?offset={offset}&limit=100')
        part = result.get('requests', [])
        rows.extend(part)
        if not result.get('hasMore'):
            return rows
        if not part:
            raise RuntimeError('Batch pagination reports more rows but returned none')
        offset += len(part)


def run_batch(client, plan, folder, repeats, timeout):
    from benchmark_backends import BASE, analyze, digest, summarize
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'report.json'
    def request(method, route, data=None):
        body = json.dumps(data).encode() if data is not None else None
        if body and len(body) > 9_500_000:
            raise ValueError('Batch request exceeds conservative 9.5 MB payload limit')
        req = Request(client.endpoint + route, data=body, method=method,
                      headers={'Authorization': f'Bearer {client.key}', 'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            # Store privately; validation details can echo source data, so do not log the body.
            (folder / 'api-error.json').write_bytes(exc.read())
            raise RuntimeError(f'Batch HTTP {exc.code}; details saved privately in api-error.json') from None
    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(path)
    if path.exists():
        report = json.loads(path.read_text())
        if report['plan_hash'] != digest(plan) or report['repeats'] != repeats:
            raise ValueError('Plan or repeat count changed; choose another output directory')
        if report.get('state') in ('CREATING', 'ADDING', 'DRAFT', 'FINALIZING'):
            raise RuntimeError('An API mutation has an uncertain outcome; inspect the batch before resuming')
    else:
        report = {'plan_hash': digest(plan), 'plan': plan, 'repeats': repeats,
                  'created': datetime.now(timezone.utc).isoformat(), 'runs': [], 'state': 'CREATING'}
        save()
        created = request('POST', '/batch', [])
        report['batch_id'] = created['id']
        report['state'] = 'DRAFT'
        save()
        print('Created batch', report['batch_id'], flush=True)
        tasks = [(rep, config, source) for rep in range(repeats) for config in plan['configs']
                 for source in config.get('sources', plan['sources'])]
        random.Random(927).shuffle(tasks)
        known_ids = set()
        for index, (rep, config, source) in enumerate(tasks):
            source = Path(source)
            options = dict(BASE, **config['options'])
            row = {'id': f'job-{index:04d}', 'config': config['name'], 'repeat': rep + 1,
                   'source': source.name, 'options': options, 'status': 'RUNNING',
                   'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            report['runs'].append(row)
            report['state'] = 'ADDING'
            save()
            # Add one child at a time so its ID is recoverable without relying on list ordering.
            request('POST', f"/batch/{quote(report['batch_id'], safe='')}/requests",
                           [{'input': {'sources': [source_for(Case('', source))],
                                       'options': options, 'target': {'kind': 'inbody'}}}])
            current = children(client, report['batch_id'])
            new = {r['id'] for r in current} - known_ids
            if len(new) != 1:
                raise RuntimeError(f'Expected exactly one new batch request ID, got {len(new)}; batch remains a draft')
            row['job_id'] = new.pop()
            known_ids.add(row['job_id'])
            report['state'] = 'DRAFT'
            save()
            print('Added', index + 1, '/', len(tasks), config['name'], source.name, flush=True)
        report['state'] = 'FINALIZING'
        save()
        client.request('POST', f"/batch/{quote(report['batch_id'], safe='')}/finalize")
        report['state'] = 'FINALIZED'
        save()
        print('Finalized', report['batch_id'], flush=True)
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        summary = client.request('GET', f"/batch/{quote(report['batch_id'], safe='')}")
        report['batch_status'] = summary
        progress = (summary.get('status'), summary.get('requestCompleted'),
                    summary.get('requestFailed'), summary.get('requestInProgress'), summary.get('requestTotal'))
        if progress != previous:
            print('Batch status/completed/failed/running/total:', progress, flush=True)
            previous = progress
        rows_by_id = {r['job_id']: r for r in report['runs']}
        for response in children(client, report['batch_id']):
            row = rows_by_id.get(response['id'])
            if row is None or row['status'] != 'RUNNING' or response['status'] not in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
                continue
            stored_response = {key: value for key, value in response.items() if key != 'input'}
            (folder / f"{row['id']}.json").write_text(json.dumps(stored_response, indent=2))
            row.update(execution_ms=response.get('executionTime', 0), worker_id=response.get('workerId'),
                       started_at=response.get('startedAt'), completed_at=response.get('completedAt'))
            try:
                validate(Case('', Path(row['source']), row['options'], check='exports'), response)
                docs = list(values(response['output'], 'json_content'))
                if len(docs) != 1:
                    raise ValueError('Expected one structured document per batch child')
                row['documents'] = analyze(response['output'], plan.get('references', {}))
                expected = plan.get('expected_pages', {}).get(row['source'])
                if expected is not None and row['documents'][0]['pages'] != expected:
                    raise ValueError('Returned page count differs from expected')
                row['document_hash'] = digest(docs)
                row['markdown_hash'] = digest(list(values(response['output'], 'md_content')))
                row['status'] = 'PASS'
            except Exception as exc:
                row.update(status='FAIL', reason=str(exc), error=response.get('error'))
            print(row['status'], row['config'], row['source'], flush=True)
        save()
        child_results_complete = (len(report['runs']) == summary.get('requestTotal') and bool(report['runs'])
                                  and all(r['status'] in ('PASS', 'FAIL') for r in report['runs']))
        if child_results_complete:
            report['all_child_results_retrieved'] = True
        if child_results_complete or summary.get('status') in ('FAILED', 'CANCELLED') or (summary.get('requestTotal', 0) > 0 and
                summary.get('requestCompleted', 0) + summary.get('requestFailed', 0) == summary['requestTotal']):
            report['state'] = 'FINISHED'
            save()
            summarize(folder)
            return int(any(r['status'] != 'PASS' for r in report['runs']))
        if time.monotonic() >= deadline:
            print('Batch still pending; saved report can be resumed without resubmitting:', path, flush=True)
            return 3
        time.sleep(10)
