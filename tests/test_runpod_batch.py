"""Exercise batching, correlation, persistence and resume without credentials/network."""
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
from urllib.parse import urlsplit

from runpod_batch import run_batch


class FakeClient:
    endpoint = 'https://example.test/v2/test'
    key = 'synthetic-test-key'

    def __init__(self):
        self.rows = []
        self.finalized = False
        self.add_shapes = []
        self.lagging_counter = False

    def request(self, method, path, data=None):
        if method == 'POST' and path == '/batch':
            assert data == []
            return {'id': 'batch-1', 'status': 'DRAFT'}
        if method == 'POST' and path.endswith('/requests'):
            self.add_shapes.append(data)
            self.rows.append({'id': 'request-' + str(len(self.rows)), 'input': data[0]['input'], 'status': 'IN_QUEUE'})
            return {}
        if path.endswith('/finalize'):
            self.finalized = True
            return {'status': 'FINALIZED'}
        if '/requests?' in path:
            rows = []
            for row in reversed(self.rows):  # Deliberately unrelated to submission order.
                result = dict(row)
                if self.finalized:
                    doc = {'origin': {'filename': 'sample.pdf'}, 'texts': [{'text': 'Synthetic document text'}],
                           'pages': {'1': {}}}
                    result.update(status='COMPLETED', executionTime=100,
                                  output={'status': 'success', 'errors': [], 'document': {
                                      'json_content': doc, 'md_content': 'Synthetic document text'}})
                rows.append(result)
            return {'requests': rows, 'hasMore': False}
        return {'status': 'FINALIZED' if self.lagging_counter else ('COMPLETED' if self.finalized else 'DRAFT'),
                'requestTotal': len(self.rows),
                'requestCompleted': max(0, len(self.rows) - int(self.lagging_counter)) if self.finalized else 0,
                'requestFailed': 0}

    def urlopen(self, request, timeout):
        path = urlsplit(request.full_url).path.removeprefix('/v2/test')
        data = json.loads(request.data) if request.data is not None else None
        return BytesIO(json.dumps(self.request(request.method, path, data)).encode())


class BatchTests(TestCase):
    def test_single_source_requests_correlate_and_resume_without_duplicates(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'sample.pdf'
            source.write_bytes(b'synthetic source bytes')
            plan = {'sources': [str(source)], 'configs': [
                {'name': 'first', 'options': {'do_ocr': False}},
                {'name': 'second', 'options': {'do_ocr': True}}], 'expected_pages': {'sample.pdf': 1}}
            client = FakeClient()
            client.lagging_counter = True
            folder = root / 'results'
            with patch('runpod_batch.urlopen', client.urlopen), patch('benchmark_backends.summarize'):
                self.assertEqual(run_batch(client, plan, folder, 2, 60), 0)
                self.assertEqual(run_batch(client, plan, folder, 2, 60), 0)
            self.assertEqual(len(client.add_shapes), 4)
            self.assertTrue(all(isinstance(body, list) and len(body) == 1 for body in client.add_shapes))
            self.assertTrue(all(len(body[0]['input']['sources']) == 1 for body in client.add_shapes))
            report = json.loads((folder / 'report.json').read_text())
            self.assertEqual(len({r['job_id'] for r in report['runs']}), 4)
            self.assertTrue(all(r['status'] == 'PASS' for r in report['runs']))
            for row in report['runs']:
                saved = json.loads((folder / (row['id'] + '.json')).read_text())
                self.assertNotIn('input', saved)

    def test_uncertain_submission_cannot_be_retried(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            plan = {'sources': [], 'configs': []}
            from benchmark_backends import digest
            (root / 'report.json').write_text(json.dumps({'plan_hash': digest(plan), 'repeats': 1,
                                                       'state': 'ADDING'}))
            with self.assertRaisesRegex(RuntimeError, 'uncertain outcome'):
                run_batch(FakeClient(), plan, root, 1, 60)


if __name__ == '__main__':
    main()
