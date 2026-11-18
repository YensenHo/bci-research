"""Gate 1 development-only SSVEP diagnosis; does not contact or download data."""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ssvep_cca import cca_rho_matrix

FS = 250


def cca_predict(trials, freqs):
    samples = trials.shape[-1]
    t = np.arange(samples) / FS
    templates = np.array([[fun(2*np.pi*h*f*t) for h in (1, 2, 3) for fun in (np.sin, np.cos)] for f in freqs])
    centered = templates - templates.mean(axis=2, keepdims=True)
    cov = np.einsum('fpi,fqi->fpq', centered, centered) / samples
    inv = np.linalg.inv(cov + np.eye(6) * 1e-6)
    return cca_rho_matrix(trials, templates, inv).argmax(axis=1)


def fit_threshold(train, quantile=.8):
    diff = np.abs(np.diff(train, axis=-1))
    threshold = np.quantile(diff, quantile, axis=(0, 2))
    return np.maximum(threshold, 1e-8)


def encode(trials, threshold):
    diff = np.diff(trials, axis=-1, prepend=trials[..., :1])
    pos = diff > threshold[None, :, None]
    neg = diff < -threshold[None, :, None]
    return np.stack((pos, neg), axis=-1).transpose(0, 2, 1, 3).reshape(len(trials), trials.shape[-1], -1).astype(np.float32)


def shuffle_events(events, seed):
    """Offline control; same time permutation for both polarities in each channel."""
    rng = np.random.default_rng(seed)
    result = np.empty_like(events)
    for trial in range(len(events)):
        for c in range(events.shape[-1] // 2):
            ix = rng.permutation(events.shape[1])
            result[trial, :, 2*c:2*c+2] = events[trial, ix, 2*c:2*c+2]
    if not np.array_equal(result.sum(axis=1), events.sum(axis=1)):
        raise AssertionError('event count changed')
    return result


class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        z = torch.sigmoid(4*x)
        return grad * 4*z*(1-z)


class StreamNet(nn.Module):
    """LIF or source-informed BD-LIF; final membrane readout, not all-spike."""
    def __init__(self, mode='lif', hidden=16):
        super().__init__()
        self.mode = mode
        self.input = nn.Linear(12, hidden, bias=False)
        self.output = nn.Linear(hidden, 40)
        if mode == 'bd':
            self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        mem = x.new_zeros((len(x), self.input.out_features))
        total = x.new_zeros(())
        spike_sum = torch.zeros_like(mem)
        for i in range(x.shape[1]):
            previous = mem
            current = (0.25 if self.mode in ('bd','signed_fixed') else 0.9)*previous + self.input(x[:, i])
            if self.mode in ('bd','signed_fixed'):
                # Source models/common.py, commit 1e7d95c; dt and theta are model parameters,
                # not a claim that dt=0.1 represents 250-Hz physical time.
                change = torch.zeros_like(current) if i == 0 else torch.abs(current-previous)/0.1
                threshold = 0.5-self.alpha*torch.tanh(change-1.5) if self.mode == 'bd' else torch.full_like(current, 0.5)
                hard = (current > threshold).to(x.dtype)-(current < -threshold).to(x.dtype)
                surrogate = torch.clamp(current, -threshold-1, -threshold+1)+torch.clamp(current, threshold-1, threshold+1)
                spike = (hard-surrogate).detach()+surrogate
                # The author code multiplies by 1-spike, including a negative spike.
                mem = current*(1-spike.detach())
            else:
                spike = Step.apply(current-1.0)
                mem = current*(1-spike.detach())
            total = total + spike.detach().abs().sum()
            spike_sum = spike_sum + spike
        self.spike_count = float(total)
        return self.output(mem + spike_sum/x.shape[1])


def predict(model, x, device):
    model.eval(); preds=[]; spikes=0.; start=time.perf_counter()
    with torch.no_grad():
        for part in torch.from_numpy(x).split(32):
            logits = model(part.to(device))
            if not torch.isfinite(logits).all():
                raise ValueError(f'nonfinite inference output for {model.mode}')
            pred = logits.argmax(1).cpu().numpy()
            if device.type == 'mps': torch.mps.synchronize()
            preds.extend(pred.tolist()); spikes += model.spike_count
    return np.asarray(preds), spikes/len(x), (time.perf_counter()-start)/len(x)


def fit(train_x, train_y, val_x, val_y, mode, seed, epochs, patience, device):
    torch.manual_seed(seed)
    model = StreamNet(mode).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(data, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best=-1.; state=None; stale=0; best_epoch=0
    for epoch in range(1, epochs+1):
        model.train()
        for x,y in loader:
            opt.zero_grad()
            loss=nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss): raise ValueError(f'nonfinite loss for {mode}')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f'nonfinite gradient for {mode}')
            opt.step()
        pred,_,_=predict(model,val_x,device); acc=float(np.mean(pred==val_y))
        if acc > best: best, state, stale, best_epoch = acc,copy.deepcopy(model.state_dict()),0,epoch
        else:
            stale+=1
            if stale>=patience: break
    model.load_state_dict(state)
    return model, {'validation_accuracy':best,'best_epoch':best_epoch,'epochs_run':epoch}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--npz',type=Path,required=True)
    p.add_argument('--freq-phase',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,11)))
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--patience',type=int,default=10)
    args=p.parse_args()
    if args.epochs<1 or args.patience<1: p.error('epochs and patience must be positive')
    with np.load(args.npz,allow_pickle=False) as d: x=np.asarray(d['eeg_1'],dtype=np.float32)
    if x.shape!=(len(args.subjects),6,40,6,250) or not np.isfinite(x).all(): raise ValueError(f'invalid eeg_1 shape/values {x.shape}')
    freqs=np.asarray(sio.loadmat(args.freq_phase)['freqs']).reshape(-1)
    if len(freqs)!=40 or not np.isfinite(freqs).all() or len(np.unique(freqs))!=40: raise ValueError('expected 40 distinct frequencies')
    result={'protocol':'same subject blocks 1-4 train, 5 val, 6 test; offline developmental',
            'subjects':args.subjects,'seed':args.seed,'device':args.device,'epochs':args.epochs,'patience':args.patience,
            'files_sha256':{z.name:hashlib.sha256(z.read_bytes()).hexdigest() for z in (args.npz,args.freq_phase)},
            'results':{},'note':'CCA full window and shuffled control are offline; BD source-informed, not a validated reproduction; no hardware energy claim.'}
    device=torch.device(args.device)
    for sid, subject in zip(args.subjects, x):
        y=np.tile(np.arange(40,dtype=np.int64),6)
        trials=subject.reshape(240,6,250)
        train_x, val_x, test_x=trials[:160],trials[160:200],trials[200:]
        train_y,val_y,test_y=y[:160],y[160:200],y[200:]
        cca=cca_predict(test_x,freqs)
        threshold=fit_threshold(train_x)
        tr,va,te=[encode(z,threshold) for z in (train_x,val_x,test_x)]
        shuffled=[shuffle_events(z,args.seed+idx) for idx,z in enumerate((tr,va,te))]
        row={'cca_accuracy':float(np.mean(cca==test_y)),'thresholds':threshold.tolist(),
             'mean_input_events':float(te.sum()/len(te)), 'models':{}}
        for name,mode,train_events,val_events,test_events in (
            ('lif','lif',tr,va,te), ('lif_shuffled_train','lif',shuffled[0],shuffled[1],shuffled[2]),
            ('signed_fixed','signed_fixed',tr,va,te), ('bd_candidate','bd',tr,va,te)):
            model,info=fit(train_events,train_y,val_events,val_y,mode,args.seed,args.epochs,args.patience,device)
            pred,spikes,seconds=predict(model,test_events,device)
            info.update(test_accuracy=float(np.mean(pred==test_y)),internal_events_per_trial=spikes,seconds_per_trial=seconds)
            if name=='lif':
                spred,_,_=predict(model,shuffled[2],device)
                info['shuffle_test_only_accuracy']=float(np.mean(spred==test_y))
            row['models'][name]=info
        result['results'][str(sid)]=row
        print('completed subject',sid,flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    pairs={
        'bd_vs_signed_fixed':('bd_candidate','signed_fixed'),
        'bd_vs_lif':('bd_candidate','lif'),
        'shuffled_train_vs_lif':('lif_shuffled_train','lif'),
    }
    result['paired_descriptive']={}
    for name,(a,b) in pairs.items():
        diffs=np.asarray([r['models'][a]['test_accuracy']-r['models'][b]['test_accuracy'] for r in result['results'].values()])
        result['paired_descriptive'][name]=dict(mean_difference_pp=float(diffs.mean()*100),
            improved=int((diffs>0).sum()),declined=int((diffs<0).sum()),tied=int((diffs==0).sum()),
            n_subjects=len(diffs),per_subject_difference_pp=(diffs*100).tolist())
    result['status']='completed'
    args.output.write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__': main()
