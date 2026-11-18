"""Summarise paired subject-level model differences without pooling seeds as people."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

VARIANTS = ('signed_fixed', 'adaptive_only', 'attention', 'attention_adaptive')
COMPARISONS = {**{v + '_vs_baseline': (v, 'baseline') for v in VARIANTS},
               'adaptive_only_vs_signed_fixed': ('adaptive_only', 'signed_fixed'),
               'attention_adaptive_vs_adaptive_only': ('attention_adaptive', 'adaptive_only')}


def describe(differences):
    d = np.asarray(differences, dtype=float)
    if d.shape != (9,) or not np.isfinite(d).all():
        raise ValueError('expected nine finite subject differences')
    return dict(mean_pp=float(d.mean()*100), sd_pp=float(d.std(ddof=1)*100),
                improved=int((d > 0).sum()), declined=int((d < 0).sum()),
                tied=int((d == 0).sum()),
                wilcoxon_p=float(wilcoxon(d).pvalue) if np.any(d) else 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024])
    args = p.parse_args()
    per_seed = {}
    for seed in args.seeds:
        all_models = {}
        reference = None
        for variant in ('baseline', *VARIANTS):
            data = json.loads((args.results_dir / f'{variant}-{seed}' / 'summary.json').read_text(encoding='utf8'))
            signature = (data['seed'], data['data_sha256'], data['warm_start_sha256'], data['protocol'])
            if data['variant'] != variant or data['seed'] != seed:
                raise ValueError(f'mismatched variant or seed: {variant} {seed}')
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError(f'mismatched data, warm start, or protocol: {variant} {seed}')
            scores = data['subject_accuracy']
            if set(scores) != {str(i) for i in range(1, 10)}:
                raise ValueError(f'invalid subject identifiers: {variant} {seed}')
            if any(scores[str(i)]['n'] != 288 for i in range(1, 10)):
                raise ValueError(f'unexpected subject trial counts: {variant} {seed}')
            all_models[variant] = np.array([scores[str(i)]['accuracy'] for i in range(1, 10)])
        per_seed[seed] = {name: all_models[a] - all_models[b] for name, (a, b) in COMPARISONS.items()}
    result = {}
    for name in COMPARISONS:
        rows = {str(seed): dict(describe(per_seed[seed][name]), subject_difference_pp=(per_seed[seed][name]*100).tolist()) for seed in args.seeds}
        avg = np.stack([per_seed[seed][name] for seed in args.seeds]).mean(axis=0)
        result[name] = dict(per_seed=rows, subject_mean_across_seeds=dict(describe(avg), subject_difference_pp=(avg*100).tolist()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
