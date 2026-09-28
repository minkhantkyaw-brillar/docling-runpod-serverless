"""Offline checks for benchmark metrics, using synthetic documents only."""
import unittest
from benchmark_backends import analyze, document_text, token_score


class BenchmarkMetrics(unittest.TestCase):
    def test_missing_and_extra_tokens_reduce_score(self):
        score = token_score('alpha beta beta 123-4', 'alpha beta extra')
        self.assertAlmostEqual(score['token_recall'], .5)
        self.assertAlmostEqual(score['token_precision'], 2 / 3)

    def test_table_values_count_once(self):
        doc = {'texts': [{'text': 'Heading'}], 'tables': [{'data': {
            'table_cells': [{'text': '123'}], 'grid': [[{'text': '123'}]]}}]}
        self.assertEqual(document_text(doc), 'Heading\n123')

    def test_timing_ignored_but_structure_and_text_changes_detected(self):
        doc = {'name': 'sample', 'origin': {'filename': 'sample.pdf'},
               'texts': [{'text': 'Company 123', 'prov': [{'bbox': {'l': 5}}]}]}
        a = analyze({'time': 1, 'document': {'json_content': doc}}, {})[0]
        b = analyze({'time': 2, 'document': {'json_content': doc}}, {})[0]
        self.assertEqual(a['json_hash'], b['json_hash'])
        doc['texts'][0]['prov'][0]['bbox']['l'] = 6
        c = analyze({'document': {'json_content': doc}}, {})[0]
        self.assertNotEqual(a['json_hash'], c['json_hash'])
        self.assertEqual(a['text_hash'], c['text_hash'])
        self.assertEqual(a['semantic_hash'], c['semantic_hash'])
        doc['texts'][0]['text'] = 'Company 128'
        d = analyze({'document': {'json_content': doc}}, {})[0]
        self.assertNotEqual(a['text_hash'], d['text_hash'])
        self.assertNotEqual(a['semantic_hash'], d['semantic_hash'])

    def test_anchors_preserve_numeric_punctuation(self):
        doc = {'origin': {'filename': 'x.pdf'}, 'texts': [{'text': 'ACME\nLIMITED 12,345'}]}
        result = analyze({'json_content': doc}, {'x.pdf': {'anchors': ['ACME LIMITED', '12,345', '12345']}})[0]
        self.assertEqual(result['anchors_found'], 2)
        self.assertEqual(result['missing_anchors'], ['12345'])


if __name__ == '__main__':
    unittest.main()
