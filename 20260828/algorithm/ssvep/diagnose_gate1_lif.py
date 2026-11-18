"""Developmental 2x2 diagnostic: raw/events × terminal-rate/temporal readout."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from gate1_ssvep import Step, encode, fit_threshold


class DiagnosticLIF(nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.readout=readout
        self.input=nn.Linear(12,16,bias=False)
        self.output=nn.Linear(160 if readout=='temporal' else 16,40)

    def forward(self,x):
        mem=x.new_zeros((len(x),16)); total=torch.zeros_like(mem); bins=[]; current_bin=torch.zeros_like(mem)
        for i in range(x.shape[1]):
            current=.9*mem+self.input(x[:,i]); spike=Step.apply(current-1.)
            mem=current*(1-spike.detach()); total=total+spike; current_bin=current_bin+spike
            if (i+1)%25==0:
                bins.append(current_bin/25); current_bin=torch.zeros_like(mem)
        features=torch.cat(bins,1) if self.readout=='temporal' else mem+total/x.shape[1]
        return self.output(features)


def run_model(x,y,readout,seed,max_epochs,patience,selection,device):
    torch.manual_seed(seed)
    model=DiagnosticLIF(readout).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loader=DataLoader(TensorDataset(torch.from_numpy(x[:160]),torch.from_numpy(y[:160])),
                      batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf') if selection=='loss' else -1.; state=None; stale=0; history=[]; best_epoch=0
    def metrics(xx,yy):
        model.eval()
        with torch.no_grad():
            logits=torch.cat([model(part.to(device)).cpu() for part in torch.from_numpy(xx).split(32)])
        if not torch.isfinite(logits).all(): raise ValueError('nonfinite logits')
        return float(nn.functional.cross_entropy(logits,torch.from_numpy(yy))),float((logits.argmax(1)==torch.from_numpy(yy)).float().mean())
    for epoch in range(1,max_epochs+1):
        model.train()
        for xx,yy in loader:
            optimizer.zero_grad(); logits=model(xx.to(device)); loss=nn.functional.cross_entropy(logits,yy.to(device))
            if not torch.isfinite(loss): raise ValueError('nonfinite loss')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('invalid gradient')
            optimizer.step()
        tl,ta=metrics(x[:160],y[:160]); vl,va=metrics(x[160:200],y[160:200])
        history.append(dict(epoch=epoch,train_loss=tl,train_accuracy=ta,validation_loss=vl,validation_accuracy=va))
        criterion=vl if selection=='loss' else va
        improved=criterion<best if selection=='loss' else criterion>best
        if improved: best,state,stale,best_epoch=criterion,copy.deepcopy(model.state_dict()),0,epoch
        else:
            stale+=1
            if stale>=patience: break
    model.load_state_dict(state)
    test_loss,test_acc=metrics(x[200:],y[200:])
    return dict(test_accuracy=test_acc,test_loss=test_loss,best_epoch=best_epoch,epochs_run=epoch,
                parameters=sum(p.numel() for p in model.parameters()),selection=selection,history=history)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,11)))
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--long-epochs',type=int,default=100);p.add_argument('--long-patience',type=int,default=25)
    args=p.parse_args()
    with np.load(args.input,allow_pickle=False) as f: data=np.asarray(f['eeg_1'],dtype=np.float32)
    if data.ndim!=5 or data.shape[1:]!=(6,40,6,250) or not np.isfinite(data).all(): raise ValueError(f'invalid shape {data.shape}')
    if min(args.subjects)<1 or max(args.subjects)>len(data): raise ValueError('subject IDs must refer to S1–S10 NPZ order')
    y=np.tile(np.arange(40,dtype=np.int64),6); device=torch.device(args.device)
    result={'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'seed':args.seed,'device':args.device,
            'protocol':'same subject blocks 1–4 train, 5 validation, 6 test; developmental',
            'warning':'Temporal readout is full-window offline; raw vs event scales differ. Long training changes stopping budget; not a clinical or hardware result.',
            'results':{}}
    for sid in args.subjects:
        trial=data[sid-1].reshape(240,6,250)
        threshold=fit_threshold(trial[:160]); events=encode(trial,threshold)
        mean=trial[:160].mean(axis=(0,2),keepdims=True)
        std=np.maximum(trial[:160].std(axis=(0,2),keepdims=True),1e-8)
        raw=((trial-mean)/std).transpose(0,2,1)
        # Match the 12 input weights: raw data occupy one lane per channel, the other lane is zero.
        raw=np.stack((raw,np.zeros_like(raw)),axis=-1).reshape(len(raw),250,12).astype('float32')
        row={'threshold':threshold.tolist(),'raw_train_mean':mean.reshape(-1).tolist(),'raw_train_std':std.reshape(-1).tolist(),'models':{}}
        for name,inputs,readout,epochs,patience,selection in (
            ('events_rate_original',events,'rate',60,10,'accuracy'),
            ('events_rate_long_acc',events,'rate',args.long_epochs,args.long_patience,'accuracy'),
            ('events_rate_long',events,'rate',args.long_epochs,args.long_patience,'loss'),
            ('events_temporal',events,'temporal',args.long_epochs,args.long_patience,'loss'),
            ('raw_rate',raw,'rate',args.long_epochs,args.long_patience,'loss'),
            ('raw_temporal',raw,'temporal',args.long_epochs,args.long_patience,'loss')):
            row['models'][name]=run_model(inputs,y,readout,args.seed,epochs,patience,selection,device)
            print('subject',sid,name,'done',flush=True)
        result['results'][str(sid)]=row
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    result['status']='completed';args.output.write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__': main()
