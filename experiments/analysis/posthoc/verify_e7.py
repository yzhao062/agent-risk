"""Post hoc E7 allowance verification.

Paired success contrasts are computed by ../paired-audit/audit_e7.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scipy.stats import beta

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOGS_DEFAULT = Path(os.environ.get("AGENT_RISK_LOGS", str(REPO / "data" / "logs"))) / "e7"
FIGURES_DEFAULT = REPO / "experiments" / "figures" / "figure_values.json"
PAPER_COMMIT = 'ee622b9f05d18a8e304f2afec31e6554a894f63c'

parser = argparse.ArgumentParser(description="Post hoc E7 allowance verification")
parser.add_argument("--logs", default=str(LOGS_DEFAULT), help="Path to logs/e7 directory")
parser.add_argument("--figures", default=str(FIGURES_DEFAULT), help="Path to figure_values.json")
parser.add_argument("--out", default=str(HERE / "verification_e7.json"), help="Output json path")
args = parser.parse_args()

ROOT = Path(args.logs)
FIGURES_PATH = Path(args.figures)


def lower(k, n, alpha):
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n-k+1))


def upper(k, n, alpha):
    return 1.0 if k == n else float(beta.ppf(1-alpha, k+1, n-k))


p = json.loads((ROOT / 'prediction_e7.json').read_text())
a = json.loads((ROOT / 'analysis_e7.json').read_text())
skip_path = ROOT / 'posthoc_skip_reference_e7.json'
if not skip_path.exists():
    skip_path = HERE / 'e7_skip_reference.json'
s = json.loads(skip_path.read_text())

figure_bytes = FIGURES_PATH.read_bytes()
assert hashlib.sha256(figure_bytes).hexdigest() == 'f754d37861276c1df37d3b32dd3442b9d8653ec3a51ef53bd9b21407e4e1a400'
f = json.loads(figure_bytes)
out = {'n': 7200, 'gates': {}, 'cost': {}}
for gate in ['1', '2']:
    pred, ana = p['gates'][gate], a['gates'][gate]
    n = 7200
    d = {}
    d['log_only'] = {'snapshot': f['fig4'][gate]['log_alone'], 'registered_json': pred['logs_only_bounds']}
    d['log_only']['counts_snapshot'] = [round(x*n) for x in f['fig4'][gate]['log_alone']]
    resolved_harm = sum(r['stop_harm'] or 0 for r in pred['refs'])
    unknown_agreement = sum(r['harm'] is None and r['kind'] == 'agreement' for r in pred['rows'])
    open_count = sum(row['kind'] == 'continuing' and ref['stop_harm'] == 0 and row['b'] < 14
                     for row, ref in zip(pred['rows'], pred['refs']))
    final_harmless = sum(row['kind'] == 'continuing' and ref['stop_harm'] == 0 and row['b'] == 14
                         for row, ref in zip(pred['rows'], pred['refs']))
    assert [resolved_harm, resolved_harm+unknown_agreement+open_count] == d['log_only']['counts_snapshot']
    d['log_only']['counts_recomputed'] = {'lower': resolved_harm, 'unknown_agreement': unknown_agreement,
                                         'open_harm_continuations': open_count, 'final_step_harmless': final_harmless}
    for alpha in [.05, .025]:
        lo, hi = d['log_only']['counts_snapshot']
        f_lo, f_hi = pred['logs_only_bounds']['p_failure_bounds']
        succ_lo, succ_hi = round((1-f_hi)*n), round((1-f_lo)*n)
        d['log_only'][str(alpha)] = {
            'harm_lower': lower(lo, n, alpha), 'harm_upper': upper(hi, n, alpha),
            'success_lower': lower(succ_lo, n, alpha), 'success_upper': upper(succ_hi, n, alpha),
            'success_count_bounds': [succ_lo, succ_hi],
            'monotone_harm_upper': upper(2481, n, alpha),
        }
    for arm in ['identified', 'live']:
        h, fail = ana[arm]['p_harm'], ana[arm]['p_failure']
        harm_lo, harm_hi = h['k'], h['k']+h['unknown']
        succ_lo = n-fail['k']-fail['unknown']
        succ_hi = n-fail['k']
        d[arm] = {'harm_k': h['k'], 'harm_unknown': h['unknown'], 'success_k_unfavorable': succ_lo}
        for alpha in [.05, .025]:
            d[arm][str(alpha)] = {
                'harm_lower': lower(harm_lo, n, alpha), 'harm_upper': upper(harm_hi, n, alpha),
                'success_lower': lower(succ_lo, n, alpha), 'success_upper': upper(succ_hi, n, alpha),
            }
        match = a['decision_example'][arm]['candidates'][int(gate)]
        assert abs(d[arm]['0.05']['harm_upper']-match['p_harm_ub95']) < 1e-12
        assert abs(d[arm]['0.05']['success_lower']-match['p_success_lb95']) < 1e-12
    d['rows'] = {}
    for kind in ['agreement', 'continuing']:
        rows = [r for r in pred['rows'] if r['kind'] == kind]
        d['rows'][kind] = {'n': len(rows), 'harm': sum(r['harm'] or 0 for r in rows),
                           'unknown_harm': sum(r['harm'] is None for r in rows),
                           'abort': sum(r['abort'] or 0 for r in rows)}
    assert len(pred['rows']) == n
    assert sum(v['harm'] for v in d['rows'].values()) == pred['identified']['p_harm']['k']
    d['rows']['decomposition'] = pred['decomposition']
    d['refs'] = {}
    for name, harm, fail in [
        ('pass_all', pred['references']['pass_all_p_harm'], pred['references']['pass_all_p_failure']),
        ('retry_as_stop', pred['references']['stop_p_harm'], pred['references']['stop_p_failure']),
        ('skip', s['gates'][gate]['skip'], pred['references']['pass_all_p_failure']),
    ]:
        d['refs'][name] = {'harm_counts': harm, 'success_k': n-fail['k']-fail['unknown'],
                          'success_note': 'skip retains logged success; conditional on the stated reading' if name == 'skip' else ''}
        for alpha in [.05, .025]:
            d['refs'][name][str(alpha)] = {
                'harm_upper_surrogate': upper(harm['k']+harm['unknown'], n, alpha),
                'success_lower_surrogate': lower(n-fail['k']-fail['unknown'], n, alpha),
            }
    cont = pred['kinds']['continuing']
    unresolved = pred['budgeted_execution']['R_executable']
    empirical_mass = cont/n
    d['restoration'] = {'continuing_share': empirical_mass, 'unresolved_harm_share': unresolved/n}
    d['restoration']['open_harm_share'] = open_count/n
    for alpha in [.05, .025]:
        u = d['identified'][str(alpha)]['harm_upper']
        l = d['identified'][str(alpha)]['success_lower']
        dh, ds = .25-u, l-.37
        d['restoration'][str(alpha)] = {
            'overall_harm_headroom': dh, 'overall_success_headroom': ds,
            'gamma_harm_empirical_continuing_mass': dh/empirical_mass,
            'gamma_harm_empirical_unresolved_mass': dh/(unresolved/n),
            'gamma_harm_empirical_open_mass': dh/(open_count/n),
            'gamma_success_empirical_continuing_mass': ds/empirical_mass,
            'gamma_harm_mass_le_1': dh, 'gamma_success_mass_le_1': ds,
        }
    d['restoration']['one_metric_joint95_with_mass'] = {}
    for metric, k_mass in [('harm', unresolved), ('success', cont)]:
        mass_u = upper(k_mass, n, .025)
        slack = (.25-d['identified']['0.025']['harm_upper']) if metric == 'harm' else (d['identified']['0.025']['success_lower']-.37)
        d['restoration']['one_metric_joint95_with_mass'][metric] = {'mass_upper975': mass_u, 'gamma': slack/mass_u}
    alpha = .05/3
    mass_u = upper(cont, n, alpha)
    dh = .25-upper(pred['identified']['p_harm']['k']+pred['identified']['p_harm']['unknown'], n, alpha)
    ds = lower(n-pred['identified']['p_failure']['k']-pred['identified']['p_failure']['unknown'], n, alpha)-.37
    d['restoration']['three_limits_joint95'] = {
        'alpha_each': alpha, 'mass_upper': mass_u,
        'overall_harm_headroom': dh, 'overall_success_headroom': ds,
        'gamma_harm': dh/mass_u, 'gamma_success': ds/mass_u,
    }
    d['gap'] = {'point_range': ana['primary']['newcombe']['point_range'],
                'newcombe95': ana['primary']['newcombe']['newcombe_paired_95'],
                'cluster95': ana['primary']['cluster']['cluster_boot_95']}
    d['cost'] = {arm: {k: v[k] for k in ['agent_calls', 'tokens', 'complete']} for arm, v in ana['execution_cost'].items() if isinstance(v,dict)}
    d['cost']['agent_calls_ratio'] = ana['execution_cost']['agent_calls_ratio']
    out['gates'][gate] = d

base = p['baseline_collection']['agent_calls']
cont = sum(out['gates'][g]['cost']['continuations']['agent_calls'] for g in ['1', '2'])
live = sum(out['gates'][g]['cost']['live']['agent_calls'] for g in ['1', '2'])
out['cost'] = {'pass_all_calls': base, 'continuation_calls': cont, 'live_calls': live, 'shared_log_ratio': (base+cont)/live}
out['source_files'] = ['logs/e7/' + name for name in ['prediction_e7.json','analysis_e7.json','posthoc_skip_reference_e7.json']]+['experiments/figures/figure_values.json']
out['paper_commit'] = PAPER_COMMIT
out['source_sha256'] = {'logs/e7/' + name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                        for name in ['prediction_e7.json','analysis_e7.json','posthoc_skip_reference_e7.json']}
out['source_sha256']['experiments/figures/figure_values.json'] = hashlib.sha256(figure_bytes).hexdigest()
Path(args.out).write_text(json.dumps(out, indent=2)+'\n')
for g, d in out['gates'].items():
    print('gate', g)
    for key in ['log_only','identified','live','restoration','gap','cost']:
        print(key, json.dumps(d[key]))
    for key, r in d['refs'].items():
        print('ref', key, json.dumps(r))
print('cost', json.dumps(out['cost']))
