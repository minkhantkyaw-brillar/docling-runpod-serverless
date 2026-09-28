#!/usr/bin/env python3
"""Re-score saved responses offline; preserve raw reports and API responses."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

from benchmark_backends import analyze, digest
from test_endpoint import values


def differences(a, b, path=''):
    if type(a) != type(b):
        yield {'path': path, 'before': str(a)[:120], 'after': str(b)[:120]}
    elif isinstance(a, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                yield {'path': path + '/' + key, 'missing': True}
            else:
                yield from differences(a[key], b[key], path + '/' + key)
    elif isinstance(a, list):
        if len(a) != len(b):
            yield {'path': path, 'length_before': len(a), 'length_after': len(b)}
        for i, (x, y) in enumerate(zip(a, b)):
            yield from differences(x, y, path + '/' + str(i))
    elif a != b:
        item = {'path': path, 'before': a, 'after': b}
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            item['absolute_delta'] = abs(a - b)
        yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--anchor-replacements', type=Path,
                        help='Private JSON mapping of corrected manual transcriptions')
    args = parser.parse_args()
    replacements = json.loads(args.anchor_replacements.read_text()) if args.anchor_replacements else {}
    groups = defaultdict(list)
    failures, pending = [], []
    for folder in args.folders:
        report = json.loads((folder / 'report.json').read_text())
        refs = json.loads(json.dumps(report['plan'].get('references', {})))
        # Correct only local scoring; every submitted request remains intact.
        for ref in refs.values():
            if 'anchors' in ref:
                ref['anchors'] = [replacements.get(a, a) for a in ref['anchors']]
        for row in report['runs']:
            if row['status'] != 'PASS':
                (pending if row['status'] == 'RUNNING' else failures).append(
                    {'stage': folder.name, 'config': row['config'], 'source': row.get('source'),
                     'status': row['status'], 'error': row.get('error', row.get('reason'))})
                continue
            response = json.loads((folder / (row['id'] + '.json')).read_text())
            doc = list(values(response['output'], 'json_content'))[0]
            metrics = analyze(response['output'], refs)[0]
            md = list(values(response['output'], 'md_content'))
            entry = dict(metrics, run_id=row['id'], job_id=row['job_id'],
                         markdown_hash=digest(md), execution_seconds=row.get('execution_ms', 0) / 1000,
                         document=doc, stage=folder.name)
            groups[(folder.name, row['config'], metrics['filename'])].append(entry)
            (folder / (row['id'] + '.md')).write_text('\n'.join(v for v in md if isinstance(v, str)))
    output = {'anchor_replacements': replacements,
              'groups': [], 'failures': failures, 'pending': pending}
    for (stage, config, filename), rows in groups.items():
        item = {'stage': stage, 'config': config, 'filename': filename, 'observations': len(rows),
                'unique_json': len({r['json_hash'] for r in rows}),
                'unique_semantic': len({r['semantic_hash'] for r in rows}),
                'unique_text': len({r['text_hash'] for r in rows}),
                'unique_markdown': len({r['markdown_hash'] for r in rows}),
                'token_f1': [r['token_f1'] for r in rows if 'token_f1' in r],
                'anchors_found': [r['anchors_found'] for r in rows], 'anchors_total': rows[0]['anchors_total'],
                'missing_anchors': [r['missing_anchors'] for r in rows],
                'tables': [r['tables'] for r in rows],
                'mean_execution_seconds': mean(r['execution_seconds'] for r in rows),
                'run_ids': [r['run_id'] for r in rows]}
        changes = []
        for a, b in combinations(rows, 2):
            if a['json_hash'] != b['json_hash']:
                delta = list(differences(a['document'], b['document']))
                changes.append({'runs': [a['run_id'], b['run_id']], 'difference_count': len(delta),
                                'bbox_only': all('/bbox/' in d['path'] for d in delta),
                                'max_numeric_delta': max((d.get('absolute_delta', 0) for d in delta), default=0),
                                'examples': delta[:12]})
        item['differences'] = changes
        output['groups'].append(item)
        print(stage, config, filename, f"runs={len(rows)} JSON={item['unique_json']} semantic={item['unique_semantic']} MD={item['unique_markdown']}",
              'F1=' + ','.join(f'{v:.4f}' for v in item['token_f1']), f"anchors={item['anchors_found']}/{item['anchors_total']}")
    args.output.write_text(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
