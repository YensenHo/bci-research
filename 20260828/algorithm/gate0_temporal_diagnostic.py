"""Gate 0: distinguish event timing/polarity order from event counts.

Synthetic research diagnostic only; no EEG, paper reproduction or hardware claim.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from lif_eegnet import SurrogateStep

STEPS = 48
SIZES = (4096, 1024, 4096)
SPLIT_SEEDS = (1001, 2002, 3003)


def make_data(task):
    """Four features are channel 0+/0- and channel 1+/1-."""
    used = set()
    splits = []
    for n, seed in zip(SIZES, SPLIT_SEEDS):
        rng = np.random.default_rng(seed + (10000 if task == 'polarity' else 0))
        labels = np.tile([0, 1], n // 2)
        rng.shuffle(labels)
        x = np.zeros((n, STEPS, 4), dtype=np.float32)
        for i, label in enumerate(labels):
            while True:
                early = rng.choice(np.arange(2, 22), 2, replace=False)
                late = rng.choice(np.arange(26, 46), 2, replace=False)
                sample = np.zeros((STEPS, 4), dtype=np.float32)
                if task == 'timing':
                    sample[early, 0 if label == 0 else 2] = 1
                    sample[late, 2 if label == 0 else 0] = 1
                elif task == 'polarity':
                    sample[early, 0 if label == 0 else 1] = 1
                    sample[late, 1 if label == 0 else 0] = 1
                else:
                    raise ValueError(task)
                key = sample.tobytes()
                if key not in used:
                    used.add(key)
                    x[i] = sample
                    break
        expected = (2, 0, 2, 0) if task == 'timing' else (2, 2, 0, 0)
        assert np.all(x.sum(axis=1) == np.asarray(expected))
        splits.append((torch.from_numpy(x), torch.from_numpy(labels.astype(np.int64))))
    return splits


class TemporalLIF(nn.Module):
    """Streaming LIF: state is local to each forward call and resets per trial."""
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, 16)
        self.output = nn.Linear(16, 2)

    def forward(self, x):
        mem = x.new_zeros((len(x), 16))
        for t in range(x.shape[1]):
            mem = 0.94 * mem + self.input(x[:, t])
            spk = SurrogateStep.apply(mem - 1.0, 4.0)
            mem = mem * (1.0 - spk)
        return self.output(mem)


class Counts(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(4, 2)

    def forward(self, x):
        return self.output(x.sum(dim=1))


class ConstantRate(nn.Module):
    """Rate-only control with the same input counts; no temporal information."""
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(4, 2)

    def forward(self, x):
        return self.output(x.mean(dim=1))


def accuracy(model, split):
    model.eval()
    x, y = split
    with torch.no_grad():
        predictions = torch.cat([model(z).argmax(dim=1) for z in x.split(256)])
    return int((predictions == y).sum()), len(y)


def wilson(correct, n):
    z = 1.959963984540054
    p = correct / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [center-half, center+half]


def train(model_type, splits, seed, shuffled=False):
    torch.manual_seed(seed)
    model = model_type()
    train_x, train_y = splits[0]
    val_x, val_y = splits[1]
    if shuffled:
        rng = np.random.default_rng(seed + 90000)
        train_y = train_y[torch.from_numpy(rng.permutation(len(train_y)))]
        val_y = val_y[torch.from_numpy(rng.permutation(len(val_y)))]
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    best, state, stale, best_epoch = -1, None, 0, 0
    for epoch in range(1, 41):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('non-finite or missing gradient')
            optimizer.step()
        correct, n = accuracy(model, (val_x, val_y))
        if correct/n > best:
            best, state, stale, best_epoch = correct/n, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
        if stale >= 8 or (best == 1.0 and not shuffled):
            break
    model.load_state_dict(state)
    # A previous call must not leak state into a later independent trial.
    model.eval()
    with torch.no_grad():
        a = model(splits[2][0][:8]); model(splits[2][0][8:16]); b = model(splits[2][0][:8])
        assert torch.equal(a, b), 'state carried across independent trials'
    true_correct, n = accuracy(model, splits[2])
    test = splits[2]
    if shuffled:
        test = (test[0], test[1][torch.from_numpy(rng.permutation(len(test[1])))])
    correct, n = accuracy(model, test)
    return dict(accuracy=correct/n, correct=correct, n=n, ci95_wilson=wilson(correct, n),
                true_label_accuracy=true_correct/n if shuffled else None,
                best_validation_accuracy=best, best_epoch=best_epoch, epochs_run=epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = dict(split_seeds=SPLIT_SEEDS, sizes=SIZES, training_seed=args.seed,
                   task_definitions='two event pairs; exact per-feature counts constant; only early/late order changes',
                   results={})
    for task in ('timing', 'polarity'):
        splits = make_data(task)
        digest = [hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest() for x,y in splits]
        vals = {}
        for name, model, shuffle in (('counts', Counts, False), ('constant_rate', ConstantRate, False),
                                     ('temporal_lif', TemporalLIF, False), ('permuted_labels_lif', TemporalLIF, True)):
            vals[name] = train(model, splits, args.seed, shuffle)
            print(task, name, vals[name], flush=True)
        temporal_ok = vals['temporal_lif']['accuracy'] >= .90
        controls_ok = all(.47 <= vals[x]['accuracy'] <= .53 for x in ('counts','constant_rate','permuted_labels_lif'))
        true_null_in_range = .47 <= vals['permuted_labels_lif']['true_label_accuracy'] <= .53
        results['results'][task] = dict(split_sha256=digest, models=vals,
                                        permuted_test_gate_pass=temporal_ok and controls_ok,
                                        true_test_single_permutation_in_range=true_null_in_range,
                                        strict_gate_pass=temporal_ok and controls_ok and true_null_in_range,
                                        gate_rule='temporal >= 0.90; count, rate, permuted-test and permuted-train/true-test all 0.47–0.53')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
