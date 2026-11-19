#!/usr/bin/env python3
"""ISRUC subject-held-out diagnostic: PSD ridge first, then signed-event LIF.

Both classifiers use the same six non-overlapping 5 s segments from each
30 s epoch; their segment scores are averaged to produce one epoch prediction.
LIF state resets per segment. This is a diagnostic implementation, not a paper
reproduction or a claim about 30 s continuous-state integration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn

FS = 100
EPOCH_SAMPLES = 3000
WINDOW_SAMPLES = 500
WINDOWS_PER_EPOCH = EPOCH_SAMPLES // WINDOW_SAMPLES
N_CLASSES = 5
SEEDS = (42, 123)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data(path: Path):
    with np.load(path, allow_pickle=False) as d:
        x = np.asarray(d["X"], dtype=np.float32)
        y = np.asarray(d["y"], dtype=np.int64)
        subj = np.asarray(d["subject"], dtype=np.int64)
    if x.ndim != 4 or x.shape[1:] != (1, 1, EPOCH_SAMPLES):
        raise ValueError(f"unexpected X shape: {x.shape}")
    x = x[:, 0, 0, :]
    if not (len(x) == len(y) == len(subj)) or not np.isfinite(x).all():
        raise ValueError("misaligned arrays or non-finite EEG")
    if set(np.unique(y)) - set(range(N_CLASSES)):
        raise ValueError("label outside fixed 5-class mapping")
    if set(np.unique(subj)) != set(range(12)):
        raise ValueError(f"expected all 12 subject IDs, got {np.unique(subj).tolist()}")
    return x, y, subj


def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def score_summary(records, model_a, model_b):
    diffs = np.asarray([r[model_a]["accuracy"] - r[model_b]["accuracy"] for r in records])
    mean = float(diffs.mean())
    rng = np.random.default_rng(20261119)
    boot = np.mean(rng.choice(diffs, (20000, len(diffs)), replace=True), axis=1)
    return {
        "n_subjects": int(len(diffs)),
        "mean_difference_pp": mean * 100,
        "subject_sd_pp": float(diffs.std(ddof=1) * 100),
        "bootstrap_95_ci_pp": [float(np.quantile(boot, .025) * 100), float(np.quantile(boot, .975) * 100)],
        "improved": int((diffs > 1e-12).sum()),
        "declined": int((diffs < -1e-12).sum()),
        "tied": int((np.abs(diffs) <= 1e-12).sum()),
        "per_subject_difference_pp": (diffs * 100).tolist(),
    }


def make_psd_features(x):
    segments = np.ascontiguousarray(x.reshape(-1, WINDOWS_PER_EPOCH, WINDOW_SAMPLES)
                                    .reshape(-1, WINDOW_SAMPLES))
    freq, power = welch(segments, fs=FS, nperseg=WINDOW_SAMPLES, noverlap=0,
                        detrend="constant", scaling="density", axis=-1)
    keep = (freq >= .5) & (freq <= 35.0)
    return np.log10(np.maximum(power[:, keep], 1e-12)).astype(np.float32)


def run_ridge(x, y, subj, data_digest, out_path):
    print("Computing six fixed 5-second log-PSD segments per epoch...", flush=True)
    features = make_psd_features(x)
    segment_y = np.repeat(y, WINDOWS_PER_EPOCH)
    all_folds = []
    subject_ids = sorted(np.unique(subj).tolist())
    for test_sid in subject_ids:
        val_sid = (test_sid + 1) % 12
        train_mask = (subj != test_sid) & (subj != val_sid)
        test_mask = subj == test_sid
        train_seg = np.repeat(train_mask, WINDOWS_PER_EPOCH)
        test_seg = np.repeat(test_mask, WINDOWS_PER_EPOCH)
        scaler = StandardScaler()
        train_f = scaler.fit_transform(features[train_seg])
        test_f = scaler.transform(features[test_seg])
        model = RidgeClassifier(alpha=10.0, class_weight="balanced")
        model.fit(train_f, segment_y[train_seg])
        start = time.perf_counter()
        scores = model.decision_function(test_f)
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
        epoch_scores = scores.reshape(-1, WINDOWS_PER_EPOCH, scores.shape[-1]).mean(axis=1)
        preds = model.classes_[epoch_scores.argmax(axis=1)]
        ridge_seconds = (time.perf_counter() - start) / max(1, int(test_mask.sum()))
        ty = y[test_mask]
        majority = int(np.bincount(y[train_mask], minlength=N_CLASSES).argmax())
        row = {
            "participant": f"P{test_sid + 1:02d}",
            "test_epochs": int(test_mask.sum()),
            "test_class_counts": np.bincount(ty, minlength=N_CLASSES).tolist(),
            "ridge_psd": metrics(ty, preds),
            "majority_train": metrics(ty, np.full(len(ty), majority)),
            "ridge_inference_seconds_per_30s_epoch": float(ridge_seconds),
            "validation_participant": f"P{val_sid + 1:02d}",
        }
        all_folds.append(row)
        print(f"ridge held-out {row['participant']}: acc={row['ridge_psd']['accuracy']:.3f}, "
              f"bacc={row['ridge_psd']['balanced_accuracy']:.3f}", flush=True)

    mean_acc = float(np.mean([r["ridge_psd"]["accuracy"] for r in all_folds]))
    mean_bacc = float(np.mean([r["ridge_psd"]["balanced_accuracy"] for r in all_folds]))
    mean_majority = float(np.mean([r["majority_train"]["accuracy"] for r in all_folds]))
    positive = mean_bacc > .25 and mean_acc > .20
    result = {
        "status": "complete",
        "data_sha256": data_digest,
        "protocol": "12-fold held-out-subject; deterministic val subject=(test+1) mod 12; 10 train subjects; fixed train-only scaler; six 5s segment scores averaged to one 30s epoch prediction",
        "baseline": "StandardScaler + RidgeClassifier(alpha=10, class_weight=balanced) on 5s log Welch PSD 0.5-35Hz; six segment scores averaged per 30s epoch; fixed before test; majority-class reference",
        "mean_ridge_accuracy": mean_acc,
        "mean_ridge_balanced_accuracy": mean_bacc,
        "mean_majority_accuracy": mean_majority,
        "positive_control_gate": {
            "pass": bool(positive),
            "rule": "mean bAcc > 25% and mean accuracy > 20% (5-class uniform chance)",
        },
        "folds": all_folds,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class SignedEventLIF(nn.Module):
    """Two polarity input lanes; reset per epoch; spike-rate/membrane readout."""
    def __init__(self, hidden=16, beta=.9, threshold=1.0):
        super().__init__()
        self.input = nn.Linear(2, hidden, bias=False)
        self.output = nn.Linear(hidden, N_CLASSES)
        self.beta = beta
        self.threshold = threshold
        self.last_rate = 0.0

    def forward(self, x):  # x [B, 500, 2] : 5 s at 100 Hz
        mem = x.new_zeros((len(x), self.input.out_features))
        spike_sum = torch.zeros_like(mem)
        for t in range(x.shape[1]):
            current = self.beta * mem + self.input(x[:, t])
            hard = (current > self.threshold).to(x.dtype)
            soft = torch.sigmoid(4.0 * (current - self.threshold))
            spike = hard + soft - soft.detach()
            mem = current * (1.0 - hard)
            spike_sum = spike_sum + spike
        if not self.training:
            self.last_rate = float((spike_sum.detach().mean() / x.shape[1]).item())
        return self.output(mem + spike_sum / x.shape[1])


def encode_events(raw_segments, threshold):
    diff = np.diff(raw_segments, axis=-1, prepend=raw_segments[..., :1])
    pos = diff > threshold
    neg = diff < -threshold
    events = np.stack((pos, neg), axis=-1).astype(np.float32)
    return events


@torch.no_grad()
def eval_lif(model, raw_epochs, y, threshold, device, batch_size=512):
    segments = np.ascontiguousarray(raw_epochs.reshape(-1, WINDOWS_PER_EPOCH, WINDOW_SAMPLES)
                                    .reshape(-1, WINDOW_SAMPLES))
    events = encode_events(segments, threshold)
    model.eval()
    scores, rates = [], []
    start = time.perf_counter()
    for first in range(0, len(events), batch_size):
        xb = torch.from_numpy(events[first:first + batch_size]).to(device)
        logits = model(xb)
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite LIF inference logits")
        scores.append(logits.cpu().numpy())
        rates.append(model.last_rate)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    scores = np.concatenate(scores).reshape(len(raw_epochs), WINDOWS_PER_EPOCH, N_CLASSES).mean(axis=1)
    return metrics(y, scores.argmax(axis=1)), scores, float(np.mean(rates)), elapsed / max(1, len(raw_epochs))


def train_lif_fold(x, y, subj, test_sid, seed, device, epochs, patience):
    val_sid = (test_sid + 1) % 12
    train_mask = (subj != test_sid) & (subj != val_sid)
    val_mask = subj == val_sid
    test_mask = subj == test_sid
    train_x, train_y = x[train_mask], y[train_mask]
    val_x, val_y = x[val_mask], y[val_mask]
    test_x, test_y = x[test_mask], y[test_mask]

    # Threshold is estimated from training-subject EEG only; test and validation labels/signals do not fit it.
    diffs = np.abs(np.diff(train_x, axis=1))
    threshold = max(float(np.quantile(diffs, .80)), 1e-8)
    del diffs
    train_segments = train_x.reshape(-1, WINDOWS_PER_EPOCH, WINDOW_SAMPLES)
    class_counts = np.bincount(train_y, minlength=N_CLASSES)
    class_weights = np.zeros(N_CLASSES, dtype=np.float32)
    present = class_counts > 0
    class_weights[present] = len(train_y) / (present.sum() * class_counts[present])

    torch.manual_seed(seed)
    model = SignedEventLIF().to(device)
    model = torch.jit.script(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    y_train_t = torch.from_numpy(train_y).long()
    best_key, best_state, best_epoch, stale = (-1.0, -float("inf")), None, 0, 0

    for epoch in range(1, epochs + 1):
        chosen = np.random.default_rng(seed + epoch + 50000 + test_sid).integers(
            0, WINDOWS_PER_EPOCH, size=len(train_y))
        selected = train_segments[np.arange(len(train_y)), chosen]
        train_events = encode_events(selected, threshold)
        ev_t = torch.from_numpy(train_events)
        order = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for ix in order.split(512):
            xb = ev_t[ix].to(device)
            yb = y_train_t[ix].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite LIF loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("missing or non-finite surrogate gradient")
            optimizer.step()

        val_metrics, val_scores, _, _ = eval_lif(model, val_x, val_y, threshold, device)
        # Use subject-held-out validation accuracy; cross entropy only breaks accuracy ties.
        val_loss = float(nn.functional.cross_entropy(
            torch.from_numpy(val_scores).to(device), torch.from_numpy(val_y).to(device)
        ).item())
        key = (val_metrics["accuracy"], -val_loss)
        if key > best_key:
            best_key, best_state, best_epoch, stale = key, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    test_metrics, _, test_rate, seconds_per_epoch = eval_lif(model, test_x, test_y, threshold, device)
    return {
        "participant": f"P{test_sid + 1:02d}",
        "validation_participant": f"P{val_sid + 1:02d}",
        "seed": seed,
        "test_epochs": int(len(test_y)),
        "test_class_counts": np.bincount(test_y, minlength=N_CLASSES).tolist(),
        "signed_event_lif": test_metrics,
        "threshold_abs_diff_q80_train_only": threshold,
        "input_event_rate_per_sample": float(encode_events(
            test_x.reshape(-1, WINDOW_SAMPLES), threshold).mean()),
        "internal_spike_rate_per_neuron_step": test_rate,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "inference_seconds_per_30s_epoch_cuda": seconds_per_epoch,
        "epoch_sec": 30,
        "window_sec": 5,
        "sample_rate_hz": FS,
    }


def audit_lif_implementation(x, y, subj, device):
    """Check sequence axes, trial reset, finite surrogate gradients on training data."""
    test_sid = int(sorted(np.unique(subj).tolist())[0])
    val_sid = (test_sid + 1) % 12
    train_mask = (subj != test_sid) & (subj != val_sid)
    train_x, train_y = x[train_mask], y[train_mask]
    threshold = max(float(np.quantile(np.abs(np.diff(train_x, axis=1)), .80)), 1e-8)
    probe_segments = train_x[0].reshape(WINDOWS_PER_EPOCH, WINDOW_SAMPLES)[:4]
    events = torch.from_numpy(encode_events(probe_segments, threshold)).to(device)
    labels = torch.full((len(events),), int(train_y[0]), dtype=torch.long, device=device)
    torch.manual_seed(31019)
    model = SignedEventLIF().to(device)
    model.train()
    loss = nn.functional.cross_entropy(model(events), labels)
    loss.backward()
    finite_grad = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if not finite_grad:
        raise RuntimeError("implementation audit failed: invalid surrogate gradient")
    model.eval()
    with torch.no_grad():
        a = model(events[:2])
        b = model(events[:2])
    reset_equal = torch.equal(a, b)
    if not reset_equal or a.shape != (2, N_CLASSES):
        raise RuntimeError("implementation audit failed: state reset/shape")
    return {
        "event_shape": list(events.shape),
        "logit_shape": list(a.shape),
        "finite_surrogate_gradients": bool(finite_grad),
        "same_trial_repeated_output_equal": bool(reset_equal),
        "spikes_observed_per_neuron_step": model.last_rate,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def run_lif(x, y, subj, data_digest, baseline_path, out_path, device, epochs, patience):
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("status") != "complete" or baseline.get("data_sha256") != data_digest:
        raise ValueError("complete baseline for exactly this data file is required before SNN")
    if not baseline["positive_control_gate"]["pass"]:
        raise RuntimeError("classical positive control failed; diagnose data/labels before any SNN run")

    implementation_audit = audit_lif_implementation(x, y, subj, device)
    print("LIF implementation audit:", json.dumps(implementation_audit), flush=True)
    all_rows = []
    for test_sid in sorted(np.unique(subj).tolist()):
        for seed in SEEDS:
            row = train_lif_fold(x, y, subj, test_sid, seed, device, epochs, patience)
            row["baseline_ridge_psd_accuracy"] = baseline["folds"][test_sid]["ridge_psd"]["accuracy"]
            all_rows.append(row)
            print(f"LIF {row['participant']} seed={seed}: acc={row['signed_event_lif']['accuracy']:.3f}, "
                  f"bacc={row['signed_event_lif']['balanced_accuracy']:.3f}, best_epoch={row['best_epoch']}",
                  flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"status": "running", "data_sha256": data_digest,
                                           "fold_runs": all_rows}, indent=2), encoding="utf-8")

    grouped = []
    for sid in sorted(np.unique(subj).tolist()):
        runs = [r for r in all_rows if r["participant"] == f"P{sid + 1:02d}"]
        lif_acc = float(np.mean([r["signed_event_lif"]["accuracy"] for r in runs]))
        ridge_acc = float(baseline["folds"][sid]["ridge_psd"]["accuracy"])
        grouped.append({"participant": f"P{sid + 1:02d}", "signed_event_lif": {"accuracy": lif_acc},
                        "ridge_psd": {"accuracy": ridge_acc}, "difference_pp": (lif_acc-ridge_acc)*100,
                        "seed_accuracies": [r["signed_event_lif"]["accuracy"] for r in runs]})

    try:
        from scipy.stats import wilcoxon
        diffs = np.asarray([r["signed_event_lif"]["accuracy"] - r["ridge_psd"]["accuracy"] for r in grouped])
        p_value = float(wilcoxon(diffs, alternative="two-sided", method="auto").pvalue) if np.any(diffs) else 1.0
    except Exception as exc:
        p_value = None
        wilcoxon_error = repr(exc)
    summary = score_summary(grouped, "signed_event_lif", "ridge_psd")
    result = {
        "status": "complete",
        "data_sha256": data_digest,
        "protocol": "12-fold held-out-subject; for each test participant, next participant is validation; 10 training participants; two seeds 42/123; no random epoch split",
        "model": "80th-percentile train-only signed first-difference event encoder at 100Hz; state-reset 16-unit LIF, beta=.9, threshold=1; surrogate-gradient training; membrane + mean-spike readout on six 5s windows averaged per 30s epoch",
        "paired_comparison": "per participant average of seed 42/123 SNN accuracy minus fixed ridge PSD accuracy; participant is the unit",
        "implementation_audit": implementation_audit,
        "paired_summary": summary,
        "wilcoxon_two_sided_p_unadjusted": p_value,
        "wilcoxon_n_participants": len(grouped),
        "gate": {
            "pass": bool(summary["improved"] >= 7 and summary["mean_difference_pp"] >= 2),
            "rule": "at least 7/12 participants improve and mean paired accuracy gain >=2pp; not a significance test",
            "resource_gate": "runtime measured on local GPU; no energy measurement; assess separately",
        },
        "folds": grouped,
        "seed_runs": all_rows,
    }
    if "wilcoxon_error" in locals():
        result["wilcoxon_error"] = wilcoxon_error
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("20260828/algorithm/isruc_data_subjects/data_subjects.npz"))
    parser.add_argument("--meta", type=Path, default=Path("20260828/algorithm/isruc_data_subjects/data_subjects_meta.json"))
    parser.add_argument("--baseline-json", type=Path, default=Path("20260828/algorithm/results/isruc_snn_loocv_20261119/baseline_ridge.json"))
    parser.add_argument("--lif-json", type=Path, default=Path("20260828/algorithm/results/isruc_snn_loocv_20261119/lif_runs.json"))
    parser.add_argument("--stage", choices=("baseline", "lif"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data_digest = sha256(args.data)
    x, y, subj = load_data(args.data)
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    if meta.get("n_subjects") != 12 or meta.get("n_total") != len(y):
        raise ValueError("preprocessing metadata does not match loaded data")
    if args.stage == "baseline":
        result = run_ridge(x, y, subj, data_digest, args.baseline_json)
        print(json.dumps({k: result[k] for k in ("mean_ridge_accuracy", "mean_ridge_balanced_accuracy",
                                                    "mean_majority_accuracy", "positive_control_gate")}, indent=2),
              flush=True)
    else:
        device = torch.device(args.device)
        result = run_lif(x, y, subj, data_digest, args.baseline_json, args.lif_json,
                         device, args.epochs, args.patience)
        print(json.dumps({"paired_summary": result["paired_summary"],
                          "wilcoxon_p": result["wilcoxon_two_sided_p_unadjusted"],
                          "gate": result["gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
