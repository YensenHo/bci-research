"""Known-noise Gaussian calibration sanity check; not EEG or a new method."""
import json
from pathlib import Path
import numpy as np

rng=np.random.default_rng(20261117);results=[]
for sigma in (0.1,1.0,3.0):
    bias=rng.normal(size=100000);signal=rng.normal(size=100000);noise=rng.normal(scale=sigma,size=100000)
    observed=signal+bias;baseline=bias+noise
    alpha=(1+sigma*sigma)/(1+2*sigma*sigma);beta=-1/(1+2*sigma*sigma)
    predictors={'no_calibration':observed/2,'full_subtraction':observed-baseline,'known_noise_conditional_mean':alpha*observed+beta*baseline}
    mse={name:float(np.mean((pred-signal)**2)) for name,pred in predictors.items()}
    theory=sigma*sigma/(1+2*sigma*sigma)
    assert abs(mse['known_noise_conditional_mean']-theory)<0.015
    results.append({'baseline_noise_sd':sigma,'mse':mse,'oracle_theoretical_mse':theory})

output={'warning':'synthetic Gaussian example, oracle knows noise variance; not EEG and not evidence of clinical or product performance','seed':20261117,'n':100000,'results':results}
Path(__file__).resolve().parents[1].joinpath('outputs','合成基线噪声_机制检查结果_20261117.json').write_text(json.dumps(output,indent=2),encoding='utf8')
print(json.dumps(results))
