"""Offline tests: no credentials, document contents, or endpoint access required."""
import base64
from io import BytesIO, StringIO
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, Mock
from zipfile import ZipFile

from test_endpoint import Case, Client, cases_for, endpoint_url, validate, source_directory, main


class RunnerTests(unittest.TestCase):
    def test_source_directory_is_required(self):
        with self.assertRaisesRegex(ValueError, "TEST_SOURCE_DIR"):
            source_directory("")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertEqual(source_directory(directory), folder.resolve())
            with patch('test_endpoint.ROOT', folder):
                self.assertEqual(source_directory('.'), folder.resolve())
            with self.assertRaisesRegex(ValueError, "existing directory"):
                source_directory(str(folder / 'missing'))

    def test_list_uses_configured_folder_without_api_or_document_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'example.PDF').touch()
            dotenv = types.ModuleType('dotenv')
            dotenv.load_dotenv = Mock()
            with patch.dict('sys.modules', {'dotenv': dotenv}), \
                    patch.dict('os.environ', {'TEST_SOURCE_DIR': directory}, clear=True), \
                    patch('sys.argv', ['test_endpoint.py', '--list', '--smoke']), \
                    patch('sys.stdout', new_callable=StringIO) as output, \
                    patch('test_endpoint.Client') as client, \
                    patch.object(Path, 'read_bytes', side_effect=AssertionError('Unexpected source read')):
                self.assertEqual(main(), 0)
                client.assert_not_called()
                self.assertIn('1 cases', output.getvalue())
                self.assertIn('example.PDF', output.getvalue())
            dotenv.load_dotenv.assert_called_once()

    def test_url(self):
        self.assertEqual(endpoint_url('https://api.runpod.ai/v2/abc/run/'), 'https://api.runpod.ai/v2/abc')
        for bad in ('', 'http://api.runpod.ai/v2/abc', 'https://secret@api.runpod.ai/v2/abc',
                    'https://api.runpod.ai/v2/abc?key=secret'):
            with self.assertRaises(ValueError):
                endpoint_url(bad)

    def test_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'a.pdf').touch()
            (folder / 'b.xlsx').touch()
            cases = cases_for(folder)
            ocr = [case for case in cases if case.name.startswith('ocr-') and case.check == 'text']
            self.assertEqual({c.options['ocr_preset'] for c in ocr},
                             {'tesserocr', 'rapidocr', 'easyocr', 'tesseract'})
            for case in ocr:
                expected = 'eng' if case.options['ocr_preset'] in {'tesserocr', 'tesseract'} else 'en'
                self.assertEqual(case.options['ocr_lang'], [expected])
            self.assertTrue(all(c.raster and c.options['force_ocr'] for c in ocr))
            self.assertEqual(len({c.name for c in cases}), len(cases))

    def test_polling_and_job_id(self):
        client = Client('https://api.runpod.ai/v2/test', 'fake', interval=.01)
        replies = [{'id': 'job', 'status': 'IN_QUEUE'}, {'status': 'IN_PROGRESS'},
                   {'status': 'COMPLETED', 'output': {}}]
        submitted = []
        with patch.object(client, 'request', side_effect=replies) as request, patch('test_endpoint.time.sleep'):
            self.assertEqual(client.run({}, submitted.append)['status'], 'COMPLETED')
            self.assertEqual(request.call_count, 3)
        self.assertEqual(submitted, ['job'])
        self.assertIsNone(client.active_job)

    def test_no_duplicate_submission_on_failure(self):
        client = Client('https://api.runpod.ai/v2/test', 'fake')
        with patch.object(client, 'request', side_effect=RuntimeError('network')) as request:
            with self.assertRaises(RuntimeError):
                client.run({}, lambda _: None)
            self.assertEqual(request.call_count, 1)

    def test_failed_jobs_are_not_success(self):
        case = Case('test', Path('x.pdf'))
        for response in ({'status': 'FAILED'}, {'status': 'TIMED_OUT'},
                         {'status': 'COMPLETED', 'output': {'error': 'broken'}},
                         {'status': 'COMPLETED', 'output': {'status': 'partial_success'}}):
            with self.assertRaises(AssertionError):
                validate(case, response)

    def test_ocr_requires_document_text(self):
        case = Case('ocr', Path('x.pdf'), raster=True)
        response = {'status': 'COMPLETED', 'output': {'document': {'md_content': '<!-- image -->',
                    'filename': 'A very long name is not OCR text.pdf'}}}
        with self.assertRaises(AssertionError):
            validate(case, response)
        response['output']['document']['md_content'] = 'This is actual extracted document text.'
        self.assertGreater(validate(case, response)['text_characters'], 20)

    def test_negative_control(self):
        case = Case('control', Path('x.pdf'), check='empty')
        response = {'status': 'COMPLETED', 'output': {'document': {'md_content': '<!-- image -->'}}}
        validate(case, response)
        response['output']['document']['md_content'] = 'Unexpected text'
        with self.assertRaises(AssertionError):
            validate(case, response)

    def test_page_range(self):
        case = Case('page', Path('x.pdf'), {'to_formats': ['json']}, check='exports', expected_page=2)
        response = {'status': 'COMPLETED', 'output': {'document': {'json_content': {'pages': {'2': {}}}}}}
        validate(case, response)
        response['output']['document']['json_content']['pages']['1'] = {}
        with self.assertRaises(AssertionError):
            validate(case, response)

    def test_expected_validation_error_not_crash(self):
        case = Case('invalid', Path('x.pdf'), check='rejected')
        validate(case, {'status': 'FAILED', 'error': 'Docling Serve HTTP 422'})
        with self.assertRaises(AssertionError):
            validate(case, {'status': 'FAILED', 'error': 'CUDA out of memory'})

    def test_zip_assets(self):
        stream = BytesIO()
        with ZipFile(stream, 'w') as archive:
            archive.writestr('doc.md', 'Text')
            archive.writestr('doc.json', '{}')
            archive.writestr('page.png', b'fake')
        response = {'status': 'COMPLETED', 'output': {'base64_string': base64.b64encode(stream.getvalue()).decode()}}
        self.assertEqual(len(validate(Case('zip', Path('x.pdf'), check='zip'), response)['archive_files']), 3)

    def test_tables_disabled(self):
        case = Case('no-tables', Path('x.pdf'), {'to_formats': ['md', 'json']}, check='no_tables')
        response = {'status': 'COMPLETED', 'output': {'document': {
            'md_content': 'This is extracted text from the document.',
            'json_content': {'tables': []}}}}
        validate(case, response)
        response['output']['document']['json_content']['tables'].append({'data': {}})
        with self.assertRaises(AssertionError):
            validate(case, response)

    def test_timeout_keeps_job_for_cancellation(self):
        client = Client('https://api.runpod.ai/v2/test', 'fake', timeout=1)
        with patch.object(client, 'request', return_value={'id': 'job', 'status': 'IN_QUEUE'}), \
                patch('test_endpoint.time.monotonic', side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                client.run({}, lambda _: None)
        self.assertEqual(client.active_job, 'job')

    def test_cancellation_targets_only_active_job(self):
        client = Client('https://api.runpod.ai/v2/test', 'fake')
        client.active_job = 'job'
        with patch.object(client, 'request', return_value={}) as request:
            client.cancel()
            request.assert_called_once_with('POST', '/cancel/job')
        self.assertIsNone(client.active_job)


if __name__ == '__main__':
    unittest.main()
