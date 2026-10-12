"""Run tasks 1 and 2 after extract_inputs.py; draw-order NPZ files are transient."""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parent
SEED, REPS, BOOT_REPS = 20261008, 500, 20000
CEILING, FLOOR = .25, .37
MODES = {'per_bound95': .05, 'approval_pair_joint95': .025}
if not (WORK / 'paired_data.npz').is_file():
    raise FileNotFoundError('Run python -B extract_inputs.py to regenerate paired_data.npz')
DATA = np.load(WORK / 'paired_data.npz')
META = json.loads((WORK / 'metadata.json').read_text())
GATES = {ex: {c: DATA[f'{ex}_{c}'] for c in ('xh','xs','yh','ys','costs','kind','unit')} for ex in META}


def rng(*parts):
    return np.random.default_rng(np.random.SeedSequence([SEED, *parts]))


@lru_cache(None)
def cp_lookup(n, alpha, upper):
    k = np.arange(n+1)
    out = np.empty(n+1)
    if upper:
        out[:-1] = beta.ppf(1-alpha, k[:-1]+1, n-k[:-1])
        out[-1] = 1
    else:
        out[1:] = beta.ppf(alpha, k[1:], n-k[1:]+1)
        out[0] = 0
    return out


def decision(bounds, e1=False):
    hu, hl, sl, su = [np.asarray(bounds[k]) for k in ('HU','HL','SL','SU')]
    approve = hu <= (.10 if e1 else CEILING)
    if not e1:
        approve &= sl >= FLOOR
    refuse = (hl > (.10 if e1 else CEILING)) | ((su < FLOOR) if not e1 else False)
    return np.where(approve, 1, np.where(refuse, -1, 0))


def shares(d):
    return {name: float(np.mean(d == code)) for name, code in [('approve',1),('refuse',-1),('undecided',0)]}


def binary(vals, upper):
    return (vals != 0 if upper else vals == 1).astype(np.int8)


def bounds_from_counts(counts, n, alpha):
    return {k: cp_lookup(n,alpha,up)[counts[k]] for k,up in [('HU',True),('HL',False),('SL',False),('SU',True)]}


def direct_bounds(g, prefix, alpha):
    counts = {k: int(binary(g[prefix+metric],up).sum()) for k,metric,up in [('HU','h',True),('HL','h',False),('SL','s',False),('SU','s',True)]}
    return bounds_from_counts(counts, len(g['xh']), alpha)


def permutations(n, stream):
    gen=rng(stream,n)
    return np.stack([gen.permutation(n) for _ in range(REPS)]).astype(np.int16)


def cumulative(g,order,prefix='y'):
    return {k: np.cumsum(binary(g[prefix+metric],up)[order],axis=1,dtype=np.int32)
            for k,metric,up in [('HU','h',True),('HL','h',False),('SL','s',False),('SU','s',True)]}


def at(counts,m):
    return {k:v[:,m-1] for k,v in counts.items()}


def costs_summary(costs):
    return dict(mean=float(np.mean(costs)),p05=float(np.percentile(costs,5)),p95=float(np.percentile(costs,95)))


def matched_and_sized():
    order=permutations(7200,1)
    np.savez_compressed(WORK/'live_draw_orders.npz',orders=order)
    cumulative_counts={ex:cumulative(GATES[ex],order) for ex in ('e7g1','e7g2')}
    cumulative_costs={ex:np.cumsum(GATES[ex]['costs'][order],axis=1,dtype=np.int32) for ex in ('e7g1','e7g2')}
    output={'matched':{},'sized':{},'targeted_full':{},'targeted_model_replication':{}}
    sizes={ex:math.floor(META[ex]['continuation_calls']*7200/META[ex]['live_calls']) for ex in ('e7g1','e7g2')}
    total_live=sum(META[e]['live_calls'] for e in sizes)
    total_cont=sum(META[e]['continuation_calls'] for e in sizes)
    pooled=math.floor(total_cont*7200/total_live)
    for mode,alpha in list(MODES.items())[:2]:
        table={}
        for name,ns in [('gate_budgets',sizes),('two_gate_equal_n',{ex:pooled for ex in sizes})]:
            dsets={}
            r=dict(units=ns,gate_shares={},expected_calls={},realized_calls={})
            for ex,m in ns.items():
                b=bounds_from_counts(at(cumulative_counts[ex],m),m,alpha)
                dsets[ex]=decision(b)
                r['gate_shares'][ex]=shares(dsets[ex])
                r['expected_calls'][ex]=m/7200*META[ex]['live_calls']
                r['realized_calls'][ex]=costs_summary(cumulative_costs[ex][:,m-1])
            r['both_live_decisions']=float(np.mean((dsets['e7g1']==-1)&(dsets['e7g2']==1)))
            r['budget']=total_cont
            table[name]=r
        output['matched'][mode]=table
        output['targeted_full'][mode]={ex:{'identified':{**direct_bounds(GATES[ex],'x',alpha),'decision':int(decision(direct_bounds(GATES[ex],'x',alpha)))},
                                                 'live':{**direct_bounds(GATES[ex],'y',alpha),'decision':int(decision(direct_bounds(GATES[ex],'y',alpha)))}}
                                           for ex in sizes}
        grid=[]
        for percent in range(1,101):
            m=percent*72
            ds={ex:decision(bounds_from_counts(at(cumulative_counts[ex],m),m,alpha)) for ex in sizes}
            both=float(np.mean((ds['e7g1']==-1)&(ds['e7g2']==1)))
            row=dict(units_per_gate=m,percent=percent,both=both,**{ex+'_match':float(np.mean(d==(-1 if ex=='e7g1' else 1))) for ex,d in ds.items()})
            grid.append(row)
        first=next(r for r in grid if r['both']>=.90)
        m=first['units_per_gate'];cost=m/7200*total_live
        output['sized'][mode]=dict(first=first,expected_calls=cost,ratio_to_cont=cost/total_cont,
                                  ratio_to_log_cont=cost/(102779+total_cont),ratio_to_full_live=cost/total_live,
                                  realized_calls=costs_summary(sum(cumulative_costs[e][:,m-1] for e in sizes)),grid=grid)
        print('Sized live',mode,first,'expected calls',round(cost,2),flush=True)
    # Multinomial parametric bootstrap: iid unit vectors retain fitted within-unit dependence across endpoints and gates.
    columns=np.stack([binary(GATES[ex][c],c=='xh') for ex in sizes for c in ('xh','xs')],axis=1)
    states,counts=np.unique(columns,axis=0,return_counts=True)
    draws=rng(3).multinomial(7200,counts/7200,size=BOOT_REPS)
    sums=draws@states
    for mode,alpha in list(MODES.items())[:2]:
        ds={}
        for j,ex in enumerate(sizes):
            counts={'HU':sums[:,2*j],'HL':sums[:,2*j],'SL':sums[:,2*j+1],'SU':sums[:,2*j+1]}
            ds[ex]=decision(bounds_from_counts(counts,7200,alpha))
        output['targeted_model_replication'][mode]={'reps':BOOT_REPS,'gate_shares':{ex:shares(d) for ex,d in ds.items()},
                                                  'both_live_decisions':float(np.mean((ds['e7g1']==-1)&(ds['e7g2']==1)))}
    return output



def json_default(x):
    if isinstance(x,np.generic):
        return x.item()
    raise TypeError(type(x))


def main():
    result=dict(label='Post hoc existing-data analysis',seed=SEED,live_reps=REPS,bootstrap_reps=BOOT_REPS,
                versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),metadata=META)
    result.update(matched_and_sized())
    live=sum(g['live_calls'] for g in META.values())
    continuation=sum(g['continuation_calls'] for g in META.values())
    result['cost']=dict(continuation_calls=continuation,full_live_calls=live,
                        ratio_to_full_live=continuation/live,log_available=True,cost_status='partial')
    (WORK/'results.json').write_text(json.dumps(result,indent=2,default=json_default)+chr(10))
    DATA.close()
    print('Saved results.json',flush=True)


if __name__=='__main__':
    main()
