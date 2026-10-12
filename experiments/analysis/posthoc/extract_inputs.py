"""Extract E7 inputs for tasks 1 and 2 of the existing-log cost analysis.

Run after the sibling decision-sweep and paired-audit extractions. All inputs
are read-only; computational hashes are pinned in source_hashes.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS_DEFAULT = Path(os.environ.get("AGENT_RISK_LOGS", str(REPO / "data" / "logs"))) / "e7"
MANIFEST_DEFAULT = REPO / "data" / "manifests" / "e7" / "manifest_e7.json"

parser = argparse.ArgumentParser(description="Extract E7 inputs for tasks 1 and 2")
parser.add_argument("--logs", default=str(LOGS_DEFAULT), help="Path to logs/e7 directory")
parser.add_argument("--manifest", default=str(MANIFEST_DEFAULT), help="Path to manifest_e7.json")
parser.add_argument("--out-dir", default=str(HERE), help="Output directory")
args, _ = parser.parse_known_args()

WORK = Path(args.out_dir)
LOGS_DIR = Path(args.logs)
MANIFEST_PATH = Path(args.manifest)

EXPECTED = json.loads((HERE / 'source_hashes.json').read_text())
INPUTS = {}
FROZEN = HERE / "inputs/frozen_harness"


def resolve_key_path(key: str) -> Path:
    if key == "docs/agent-risk-modeling/agent-io-archive-2026-10-07/r14/decision-sweep/decision_map_e7.py":
        return FROZEN / "decision_map_e7.py"
    if key == "docs/agent-risk-modeling/agent-io-archive-2026-10-07/r14/decision-sweep/paired_units.jsonl":
        return HERE / "inputs/paired_units.jsonl"
    if key == "docs/agent-risk-modeling/agent-io-archive-2026-10-07/r14/paired-audit/audit_e7.py":
        return HERE / "inputs/audit_e7.py"
    if key == "docs/agent-risk-modeling/agent-io-archive-2026-10-07/r14/paired-audit/paired_units.csv":
        return HERE / "inputs/paired_units.csv"
    if key == "experiments/agent-risk-harness/e7/manifest_e7.json":
        return MANIFEST_PATH
    if key == "experiments/agent-risk-harness/agent_risk_harness/policy.py":
        return REPO / "experiments/harness/agent_risk_harness/policy.py"
    if key.startswith("experiments/agent-risk-harness/logs/e7/"):
        return LOGS_DIR / key.removeprefix("experiments/agent-risk-harness/logs/e7/")
    if key.startswith("experiments/agent-risk-harness/"):
        name = key.removeprefix("experiments/agent-risk-harness/")
        if (FROZEN / name).exists():
            return FROZEN / name
        return REPO / "experiments/analysis" / name
    return REPO / key


def read(path, role, expected=None):
    path = Path(path)
    key = None
    for k in EXPECTED:
        if resolve_key_path(k).resolve() == path.resolve():
            key = k
            break
    if key is None:
        try:
            key = path.relative_to(REPO).as_posix()
        except ValueError:
            key = path.name
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    pinned = EXPECTED[key]
    assert sha == pinned['sha256'] and len(data) == pinned['bytes'], (key, sha, pinned['sha256'])
    if expected is not None:
        assert sha == expected, (key, sha, expected)
    INPUTS[key] = dict(sha256=sha, bytes=len(data), role=role)
    return data


def js(path, role, expected=None):
    return json.loads(read(path, role, expected))


def module(path, name):
    read(path, 'Read-only registered extraction definitions')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def encode(x):
    return np.array([2 if v is None else int(v) for v in x], dtype=np.int8)


def records(paths, role, condition=None):
    scored, raw = {}, {}
    for path, expected in paths:
        for line in read(path, role, expected).splitlines():
            r = json.loads(line)
            if condition and r.get('condition') != condition:
                continue
            eid = r.get('episode_id')
            raw.setdefault(eid, r)
            if r.get('scored') and r.get('per_step') is not None:
                scored.setdefault(eid, r)
    return scored, raw


def main():
    read(resolve_key_path('docs/agent-risk-modeling/agent-io-archive-2026-10-07/r14/decision-sweep/decision_map_e7.py'), 'Source of the archived paired extraction')
    read(resolve_key_path('docs/agent-risk-modeling/agent-io-archive-2026-10-07/r14/paired-audit/audit_e7.py'), 'Source of the archived paired extraction')
    result, metadata = {}, {}
    e1_path = resolve_key_path('experiments/agent-risk-harness/analyze_e1.py')
    e1 = module(e1_path, 'registered_e1')
    pred = js(LOGS_DIR / 'prediction_e7.json', 'Frozen identified per-unit predictions',
              'a3bccf79de31846adfb037155d4f03017fad537fafabeacad3681fc6086b1bd7')
    analysis = js(LOGS_DIR / 'analysis_e7.json', 'Registered analysis, counts and costs')
    man = js(MANIFEST_PATH, 'Registered task-and-seed unit mapping')
    read(REPO / 'experiments/analysis/analyze_e7.py', 'Registered E7 analysis definitions', pred['analysis_sha256'])
    assert pred['analysis_sha256'] == analysis['analysis_sha256']
    assert INPUTS['experiments/agent-risk-harness/e7/manifest_e7.json']['sha256'] == pred['manifest_sha256'] == analysis['manifest_sha256']
    assert len(man['units']) == pred['n_units'] == analysis['n_units'] == 7200
    data = read(resolve_key_path('docs/agent-risk-modeling/agent-io-archive-2026-10-07/r14/paired-audit/paired_units.csv'), 'Archived complete E7 paired extraction')
    cached = list(csv.DictReader(data.decode().splitlines()))
    sparse = [json.loads(line) for line in read(resolve_key_path('docs/agent-risk-modeling/agent-io-archive-2026-10-07/r14/decision-sweep/paired_units.jsonl'),
                                              'Independent E7 live extraction').splitlines()]
    assert len(cached) == len(sparse) == 14400
    sparse = {(r['gate'], r['unit_id']): r for r in sparse}
    paths = [(LOGS_DIR / v["path"], v['sha256']) for v in analysis['live_inputs']]
    assert len(paths) == 150
    scored, raw = records(paths, 'Hash-checked E7 live records, both gates', 'live')
    read(REPO / 'experiments/harness/agent_risk_harness/policy.py', 'Checker used by recompute_flags')
    e1.apm.recompute_flags(list(scored.values()))
    for gate in (1, 2):
        key = f'e7g{gate}'
        rows = [r for r in cached if int(r['gate']) == gate]
        frozen = pred['gates'][str(gate)]['rows']
        assert len(rows) == len(frozen) == 7200
        for r, f, u in zip(rows, frozen, man['units']):
            assert r['unit_id'] == f['unit_id'] == u['unit_id']
            assert (None if r['identified_harm']=='' else int(r['identified_harm'])) == f['harm']
            assert (None if r['identified_success']=='' else int(r['identified_success'])) == (None if f['failure'] is None else 1-f['failure'])
            eid = u['episode_ids'][f'live{gate}']
            rr = scored.get(eid)
            h = e1.arl.shift_harm(rr, 0, None, True, False) if rr else None
            s = 1-int(rr['loss']['failure']) if rr else None
            assert h == sparse[(gate,r['unit_id'])]['live_harm'] == (None if r['live_harm']=='' else int(r['live_harm']))
            assert s == sparse[(gate,r['unit_id'])]['live_success'] == (None if r['live_success']=='' else int(r['live_success']))
            assert int(raw[eid]['e1_cost']['agent_calls'] or 0) == int(r['live_agent_calls'])
        def col(name):
            return encode([None if r[name]=='' else int(r[name]) for r in rows])
        result[key] = dict(xh=col('identified_harm'), xs=col('identified_success'), yh=col('live_harm'), ys=col('live_success'),
                           kind=np.array([r['kind'] for r in rows]), costs=np.array([int(r['live_agent_calls']) for r in rows]),
                           unit=np.array([r['unit_id'] for r in rows]))
        g = result[key]
        for src, prefix in (('identified', 'x'), ('live', 'y')):
            for metric, c in (('p_harm','h'), ('p_failure','s')):
                vals = g[prefix+c]
                known = int(np.sum(vals==1)) if c=='h' else int(np.sum(vals==0))
                unknown = int(np.sum(vals==2))
                assert known == analysis['gates'][str(gate)][src][metric]['k']
                assert unknown == analysis['gates'][str(gate)][src][metric]['unknown']
        cost = analysis['gates'][str(gate)]['execution_cost']
        live_calls = int(cost['live']['agent_calls'])
        continuation_calls = int(cost['continuations']['agent_calls'])
        assert live_calls == (102631,102244)[gate-1] == g['costs'].sum()
        assert continuation_calls == (8281,23425)[gate-1]
        assert analysis['baseline_collection']['agent_calls'] == 102779
        metadata[key] = dict(N=7200, continuation_calls=continuation_calls, live_calls=live_calls,
                             shared_log_calls=102779, cost_status='partial', serving='default kernels',
                             declared_harm_ceiling=.25, declared_success_floor=.37)
    visited = set()
    def walk(m):
        if id(m) in visited:
            return
        visited.add(id(m))
        path = getattr(m, '__file__', None)
        if path and (str(path).startswith(str(FROZEN)) or str(path).startswith(str(REPO / 'experiments/analysis'))) and str(path).endswith('.py'):
            read(path, 'Imported read-only registered extraction dependency')
            for v in vars(m).values():
                if hasattr(v, '__file__'):
                    walk(v)
    walk(e1)
    assert INPUTS.keys() == EXPECTED.keys(), f'Pinned computational input coverage differs: missing {set(EXPECTED.keys()) - set(INPUTS.keys())}'
    arrays = {f'{ex}_{col}': val for ex, g in result.items() for col, val in g.items()}
    np.savez_compressed(WORK / 'paired_data.npz', **arrays)
    with (WORK / 'paired_data.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment','unit','kind','identified_harm','identified_success','live_harm','live_success','live_agent_calls'])
        for ex, g in result.items():
            for i in range(len(g['xh'])):
                writer.writerow([ex,g['unit'][i],g['kind'][i],*[None if g[c][i]==2 else int(g[c][i]) for c in ('xh','xs','yh','ys')],int(g['costs'][i])])
    (WORK / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (WORK / 'input_manifest.json').write_text(json.dumps(INPUTS, indent=2)+'\n')
    print('Validated both E7 paired rates; E7 cache equals hash-checked raw live records.', flush=True)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
