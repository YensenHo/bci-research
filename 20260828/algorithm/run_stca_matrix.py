"""Run one seed's cross-session variants; use separate processes for parallel seeds."""
import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--data-dir', type=Path, required=True)
p.add_argument('--results-dir', type=Path, required=True)
p.add_argument('--seed', type=int, required=True)
args = p.parse_args()
trainer = Path(__file__).resolve().with_name('train_eeg_tcnet_stca.py')
for variant in ('relu', 'baseline', 'signed_fixed', 'adaptive_only', 'attention', 'attention_adaptive'):
    out = args.results_dir / f'{variant}-{args.seed}'
    if (out / 'summary.json').exists() and (out / 'model.pth').exists():
        raise SystemExit(f'refusing to overwrite existing result: {out}')
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', str(trainer), '--data-dir', str(args.data_dir), '--output-dir', str(out), '--variant', variant, '--seed', str(args.seed)]
    if variant != 'relu':
        cmd += ['--warm-start', str(args.results_dir / f'relu-{args.seed}' / 'model.pth')]
    print(f'start {variant} {args.seed}', flush=True)
    with (out / 'run.log').open('w', encoding='utf8') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
