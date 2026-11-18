"""Generate artificial sinusoids for pipeline smoke testing, never scientific results."""
import argparse
from pathlib import Path
import numpy as np
from scipy.io import savemat

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(8);frequencies=np.linspace(8,15.8,40);t=np.arange(250)/250
data=np.empty((1,6,40,6,250),dtype=np.float32)
for block in range(6):
    for target in range(40):
        data[0,block,target]=np.array([np.sin(2*np.pi*frequencies[target]*t+c*.15)+rng.normal(0,.05,250) for c in range(6)])
np.savez_compressed(args.out/'demo.npz',eeg_1=data)
savemat(args.out/'Freq_Phase.mat',{'freqs':frequencies})
