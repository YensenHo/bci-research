"""Exploratory SSVEP A/B/C/D comparisons. C explicitly receives target frequencies."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from gate1_ssvep import Step, encode, fit_threshold

FS, LENGTH, INPUTS = 250, 250, 12


def synthetic_data(seed):
    """Distinct split seeds; per-trial independent phase, amplitude and noise."""
    frequencies=np.array([8.,10.,12.,14.]); splits=[]; seen=set()
    for n,split_seed in ((512,11001),(256,22002),(1024,33003)):
        rng=np.random.default_rng(split_seed)
        y=np.tile(np.arange(4),n//4); rng.shuffle(y)
        t=np.arange(LENGTH)/FS; x=np.empty((n,6,LENGTH),dtype=np.float32)
        for i,label in enumerate(y):
            phase=rng.uniform(0,2*np.pi); amp=rng.uniform(.7,1.3)
            channel_offset=rng.uniform(-.2,.2,6)
            x[i]=amp*np.sin(2*np.pi*frequencies[label]*t[None,:]+phase+channel_offset[:,None])+rng.normal(0,.15,(6,LENGTH))
            key=hashlib.sha256(x[i].tobytes()).digest()
            if key in seen: raise ValueError('duplicate sample across splits')
            seen.add(key)
        splits.append((x,y.astype(np.int64)))
    return frequencies,splits


def select_frequencies(freqs, task):
    freqs=np.asarray(freqs,dtype=float).ravel()
    if len(freqs)!=40 or len(np.unique(freqs))!=40 or not np.isfinite(freqs).all(): raise ValueError('invalid frequency table')
    if task=='40': return np.arange(40),freqs
    indices=[]
    for value in (8.,10.,12.,14.):
        found=np.flatnonzero(np.isclose(freqs,value,atol=1e-4,rtol=0))
        if len(found)!=1: raise ValueError(f'frequency {value} is not unique')
        indices.append(int(found[0]))
    return np.array(indices),freqs[indices]


class RateNet(nn.Module):
    def __init__(self,hidden,classes):
        super().__init__(); self.input=nn.Linear(INPUTS,hidden,bias=False); self.output=nn.Linear(hidden,classes)
    def forward(self,x):
        inputs=self.input(x); mem=x.new_zeros((len(x),self.input.out_features)); total=torch.zeros_like(mem)
        for t in range(x.shape[1]):
            current=.9*mem+inputs[:,t]; spike=Step.apply(current-1.); mem=current*(1-spike.detach()); total=total+spike
        self.internal_spikes=float(total.detach().sum()); return self.output(mem+total/x.shape[1])


class FrequencyNet(nn.Module):
    """Offline quadrature magnitude readout on spikes; known target-frequency prior."""
    def __init__(self,frequencies):
        super().__init__(); self.input=nn.Linear(INPUTS,16,bias=False)
        frequencies=torch.as_tensor(frequencies,dtype=torch.float32)
        t=torch.arange(LENGTH,dtype=torch.float32)/FS
        self.register_buffer('cosine',torch.cos(2*torch.pi*torch.outer(t,frequencies)))
        self.register_buffer('sine',torch.sin(2*torch.pi*torch.outer(t,frequencies)))
        self.w=nn.Parameter(torch.empty(len(frequencies),16,2)); self.bias=nn.Parameter(torch.zeros(len(frequencies)))
        nn.init.normal_(self.w,std=.1)

    def forward(self,x):
        if x.shape[1]!=LENGTH: raise ValueError('frequency readout requires 250 samples')
        inputs=self.input(x); mem=x.new_zeros((len(x),16)); spikes=[]
        for t in range(x.shape[1]):
            current=.9*mem+inputs[:,t]; spike=Step.apply(current-1.); mem=current*(1-spike.detach()); spikes.append(spike)
        s=torch.stack(spikes,dim=1); self.internal_spikes=float(s.detach().sum())
        re=torch.einsum('bth,tk->bkh',s,self.cosine)/LENGTH
        im=torch.einsum('bth,tk->bkh',s,self.sine)/LENGTH
        amplitude=torch.sqrt(re*re+im*im+1e-8)*10
        features=torch.stack((amplitude,amplitude*amplitude),dim=-1)
        return torch.einsum('bkhd,khd->bk',features,self.w)+self.bias


def build_models(freqs,seed):
    classes=len(freqs)
    torch.manual_seed(seed); c=FrequencyNet(freqs)
    count_c=sum(p.numel() for p in c.parameters())
    width_d=max(1,round((count_c-classes)/(INPUTS+classes)))
    models={}
    for name,width in [('A',16),('B',64),('D',width_d)]:
        torch.manual_seed(seed);models[name]=RateNet(width,classes)
    models['C']=c
    return models


def score(model,x,y,device):
    model.eval(); predictions=[]; loss=0.; spikes=0.; start=time.perf_counter()
    with torch.no_grad():
        for begin in range(0,len(x),128):
            xb=torch.from_numpy(x[begin:begin+128]).to(device);yb=torch.from_numpy(y[begin:begin+128]).to(device)
            logits=model(xb)
            if not torch.isfinite(logits).all(): raise ValueError('nonfinite inference')
            loss+=float(nn.functional.cross_entropy(logits,yb,reduction='sum'));spikes+=model.internal_spikes
            predictions.append(logits.argmax(1).cpu().numpy())
    if device.type=='mps': torch.mps.synchronize()
    return dict(loss=loss/len(x),accuracy=float(np.mean(np.concatenate(predictions)==y)),
                internal_spikes_per_trial=spikes/len(x),seconds_per_trial=(time.perf_counter()-start)/len(x))


def train(model,splits,seed,epochs,patience,device):
    x,y=splits[0]; vx,vy=splits[1]; tx,ty=splits[2]
    torch.manual_seed(seed);model=model.to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');best_state=None;stale=0;history=[];best_epoch=0
    for epoch in range(1,epochs+1):
        model.train()
        for xb,yb in loader:
            opt.zero_grad();logits=model(xb.to(device));loss=nn.functional.cross_entropy(logits,yb.to(device))
            if not torch.isfinite(loss): raise ValueError('nonfinite train loss')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('invalid gradient')
            opt.step()
        tr=score(model,x,y,device);va=score(model,vx,vy,device)
        history.append(dict(epoch=epoch,train_accuracy=tr['accuracy'],train_loss=tr['loss'],validation_accuracy=va['accuracy'],validation_loss=va['loss']))
        if va['loss']<best: best=va['loss'];best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:
            stale+=1
            if stale>=patience: break
    model.load_state_dict(best_state)
    return dict(test=score(model,tx,ty,device),best_epoch=best_epoch,epochs_run=epoch,
                parameters=sum(p.numel() for p in model.parameters()),history=history)


def prepare(splits):
    threshold=fit_threshold(splits[0][0])
    return [(encode(x,threshold),y) for x,y in splits],threshold


def run_synthetic(args):
    freqs,raw=synthetic_data(args.seed);splits,threshold=prepare(raw);device=torch.device(args.device)
    result={'task':'synthetic_4','frequencies':freqs.tolist(),'split_seeds':[11001,22002,33003],
            'split_sizes':[len(y) for x,y in raw], 'train_seed':args.seed,'threshold':threshold.tolist(),
            'mean_input_events_per_trial':{name:float(x.sum()/len(x)) for name,(x,y) in zip(('train','validation','test'),splits)},
            'split_sha256':[hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest() for x,y in raw],
            'frequency_prior':'C uses target frequencies in quadrature templates','models':{},'permutations':[],'status':'running'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    untrained=build_models(freqs,args.seed)['C'].to(device)
    result['C_untrained_test']=score(untrained,splits[2][0],splits[2][1],device)
    for name,model in build_models(freqs,args.seed).items():
        result['models'][name]=train(model,splits,args.seed,args.epochs,args.patience,device)
        print('synthetic',name,result['models'][name]['test']['accuracy'],flush=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    for i in range(args.permutations):
        rng=np.random.default_rng(80000+i)
        shuffled=[(x,y[rng.permutation(len(y))] if j<2 else y) for j,(x,y) in enumerate(splits)]
        model=build_models(freqs,args.seed+i+1)['C']
        outcome=train(model,shuffled,args.seed+i+1,args.epochs,args.patience,device)
        result['permutations'].append(dict(seed=80000+i,accuracy_against_true_test=outcome['test']['accuracy'],best_epoch=outcome['best_epoch']))
        print('permutation',i,outcome['test']['accuracy'],flush=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


def run_real(args):
    if args.npz is None or args.freq_phase is None: raise ValueError('real requires --npz and --freq-phase')
    for path,expected in ((args.npz,args.expected_npz_sha256),(args.freq_phase,args.expected_freq_sha256)):
        if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'input SHA-256 mismatch: {path}')
    with np.load(args.npz,allow_pickle=False) as f: data=np.asarray(f['eeg_1'],dtype=np.float32)
    if data.ndim!=5 or data.shape[1:]!=(6,40,6,250) or not np.isfinite(data).all(): raise ValueError(f'invalid EEG shape {data.shape}')
    if len(set(args.subjects))!=len(args.subjects) or min(args.subjects)<1 or max(args.subjects)>len(data): raise ValueError('subject IDs must be unique and refer to NPZ ordered S1..Sn')
    ix,freqs=select_frequencies(loadmat(args.freq_phase)['freqs'],args.task)
    result={'task':f'real_{args.task}','seed':args.seed,'subjects':args.subjects,'target_indices_zero_based':ix.tolist(),
            'frequencies':freqs.tolist(),'frequency_prior':'C uses target frequencies in quadrature templates',
            'protocol':'same-subject blocks 1-4 train, 5 validation, 6 test; developmental and offline',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.npz,args.freq_phase)},'results':{}}
    for sid in args.subjects:
        subject=data[sid-1][:,ix,:,:];classes=len(ix)
        raw=[(subject[a:b].reshape(-1,6,LENGTH),np.tile(np.arange(classes,dtype=np.int64),b-a)) for a,b in ((0,4),(4,5),(5,6))]
        splits,threshold=prepare(raw);row={'threshold':threshold.tolist(),'mean_input_events':float(splits[2][0].sum()/len(splits[2][0])),'models':{}}
        untrained=build_models(freqs,args.seed)['C'].to(torch.device(args.device))
        row['C_untrained_test']=score(untrained,splits[2][0],splits[2][1],torch.device(args.device))
        for name,model in build_models(freqs,args.seed).items():
            row['models'][name]=train(model,splits,args.seed,args.epochs,args.patience,torch.device(args.device))
            print('subject',sid,name,'test',row['models'][name]['test']['accuracy'],flush=True)
        result['results'][str(sid)]=row
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['synthetic','real'],required=True)
    p.add_argument('--task',choices=['4','40'],default='40');p.add_argument('--npz',type=Path);p.add_argument('--freq-phase',type=Path)
    p.add_argument('--expected-npz-sha256');p.add_argument('--expected-freq-sha256')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,11)));p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=25);p.add_argument('--permutations',type=int,default=20)
    args=p.parse_args()
    if args.epochs<1 or args.patience<1 or args.permutations<0: p.error('invalid budget')
    torch.set_num_threads(2);result=run_synthetic(args) if args.mode=='synthetic' else run_real(args)
    result.update(device=args.device,epochs=args.epochs,patience=args.patience,status='completed')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__':main()
