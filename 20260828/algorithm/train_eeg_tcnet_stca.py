"""Development comparison on prepared BCIC IV 2a NPZ, cross-session only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from eeg_tcnet_stca import EEGTCNetAttention


def data(path):
    with np.load(path, allow_pickle=False) as d:
        x = np.asarray(d['X'], dtype=np.float32)
        y = np.asarray(d['y'], dtype=np.int64).reshape(-1)
    if x.shape != (len(y), 1, 22, 500) or not np.isfinite(x).all() or not np.isin(y, range(4)).all():
        raise ValueError(f'invalid shape, labels, or values: {path}')
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def accuracy(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += int((pred == y).sum()); total += len(y)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--variant', required=True, choices=['relu', 'baseline', 'attention', 'attention_adaptive', 'adaptive_only', 'signed_fixed'])
    p.add_argument('--warm-start', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--threshold', type=float, default=1.0)
    p.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    args = p.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1: p.error('positive training parameters required')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_path, test_path = args.data_dir/'train.npz', args.data_dir/'eval.npz'
    train, test = data(train_path), data(test_path)
    n_val = max(1, int(len(train) * .15)); n_train = len(train) - n_val
    train, val = random_split(train, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    if (args.variant == 'relu') != (args.warm_start is None):
        p.error('relu trains from scratch; other variants require --warm-start')
    model = EEGTCNetAttention(attention=args.variant in ('attention','attention_adaptive'), adaptive=args.variant in ('attention_adaptive', 'adaptive_only'), signed=args.variant == 'signed_fixed', activation='relu' if args.variant == 'relu' else 'lif', threshold=args.threshold, tau=2.0, timesteps=8, alpha=4.0)
    if args.warm_start is not None:
        ck = torch.load(args.warm_start, map_location='cpu', weights_only=False)
        weights = ck.get('state_dict', ck)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected or any(not name.startswith('tcn.attention.') for name in missing):
            raise ValueError(f'unsupported warm start: missing={missing}, unexpected={unexpected}')
    device = torch.device(args.device); model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = -1.; best_state = None; best_epoch = 0; stale = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y in loader:
            opt.zero_grad(); loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss): raise ValueError('non-finite loss')
            loss.backward(); opt.step()
        schedule.step(); score = accuracy(model, val_loader, device)
        print(f'epoch={epoch} val_accuracy={score:.5f}',flush=True)
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best_state)
    score = None if args.variant == 'relu' else accuracy(model, test_loader, device)
    subject_accuracy = None
    if args.variant != 'relu':
        with np.load(test_path, allow_pickle=False) as d:
            if 'subject' in d.files:
                subject_ids = np.asarray(d['subject']).reshape(-1)
                if len(subject_ids) != len(test):
                    raise ValueError('subject identifiers do not match test trials')
                subject_accuracy = {}
                for subject in np.unique(subject_ids):
                    indices = np.flatnonzero(subject_ids == subject).tolist()
                    subset = torch.utils.data.Subset(test, indices)
                    subject_accuracy[str(int(subject))] = dict(n=len(indices), accuracy=accuracy(model, DataLoader(subset, batch_size=args.batch_size), device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(variant=args.variant, seed=args.seed, device=args.device, threshold=args.threshold, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, protocol='cross-session, same subjects; random trial validation', validation_accuracy=best, best_epoch=best_epoch, test_accuracy=score, subject_accuracy=subject_accuracy, train_trials=n_train, validation_trials=n_val, test_trials=len(test), parameters=sum(p.numel() for p in model.parameters()), data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path,test_path)}, warm_start_sha256=hashlib.sha256(args.warm_start.read_bytes()).hexdigest() if args.warm_start else None)
    (args.output_dir/'summary.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    torch.save({'state_dict':best_state, 'summary':meta},args.output_dir/'model.pth')
    print(json.dumps(meta,indent=2))

if __name__ == '__main__':
    main()
