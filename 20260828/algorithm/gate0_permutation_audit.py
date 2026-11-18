"""Exploratory repeated label-permutation audit on the fixed Gate 0 splits."""
import argparse
import json
from pathlib import Path
import torch
from gate0_temporal_diagnostic import make_data, train, TemporalLIF

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--runs', type=int, default=20)
args = parser.parse_args()
torch.set_num_threads(2)
results = {}
for task in ('timing', 'polarity'):
    splits = make_data(task)
    results[task] = {}
    for seed in range(1, args.runs + 1):
        results[task][seed] = train(TemporalLIF, splits, seed, shuffled=True)
        print(task, seed, results[task][seed]['true_label_accuracy'], flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=2), encoding='utf8')
