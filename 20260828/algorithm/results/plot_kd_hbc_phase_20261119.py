from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "蒸馏抗噪_信噪比相图_HBC复核_20261119.png"

def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))

def paired(prefix, sigmas, field):
    vals=[]
    for s in sigmas:
        items=[]
        for seed in (42,123):
            suffix="_s123" if seed==123 else ""
            name=f"{prefix}_data_4class_cross_n{s}{suffix}.json"
            items.append(read(name)[field])
        vals.append(items)
    return np.asarray(vals,dtype=float)

sigmas=[0.5,1.0,1.5,2.0,3.0]
awgn_total=paired("sign_check",[f"{s:.1f}" for s in sigmas],"gain_30pct_pp")
awgn_k=paired("kd_decomposition",["1.0","1.5","2.0"],"K_pp")
awgn_r=paired("kd_decomposition",["1.0","1.5","2.0"],"R_pp")
scenarios=["quant","burst","impulse","full"]
hbc_total=[]; hbc_k=[]; hbc_r=[]
for sc in scenarios:
    hbc_total.append([read(f"sign_check_data_4class_cross_hbc_{sc}1.0.json")["gain_30pct_pp"],
                      read(f"sign_check_data_4class_cross_hbc_{sc}1.0_s123.json")["gain_30pct_pp"]])
    hbc_k.append([read(f"kd_decomposition_data_4class_cross_hbc_{sc}1.0.json")["K_pp"],
                  read(f"kd_decomposition_data_4class_cross_hbc_{sc}1.0_s123.json")["K_pp"]])
    hbc_r.append([read(f"kd_decomposition_data_4class_cross_hbc_{sc}1.0.json")["R_pp"],
                  read(f"kd_decomposition_data_4class_cross_hbc_{sc}1.0_s123.json")["R_pp"]])
hbc_total=np.asarray(hbc_total); hbc_k=np.asarray(hbc_k); hbc_r=np.asarray(hbc_r)
full15=read("kd_decomposition_data_4class_cross_hbc_full1.5.json")

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,axs=plt.subplots(1,2,figsize=(13,5.1),constrained_layout=True)
ax=axs[0]
mean=awgn_total.mean(1); lo=awgn_total.min(1); hi=awgn_total.max(1)
ax.plot(sigmas,mean,"o-",color="#2368a0",lw=2,label="AWGN mean (seeds 42, 123)")
ax.fill_between(sigmas,lo,hi,color="#2368a0",alpha=.15,label="AWGN seed range")
for i,sc in enumerate(scenarios):
    x=1.0+(i-1.5)*.045
    vals=hbc_total[i]
    ax.scatter([x,x],vals,s=45,marker="D",color=f"C{i+1}",label=f"HBC {sc} (seed 42/123)")
    ax.plot([x,x],vals,color=f"C{i+1}",alpha=.55,lw=1)
ax.axhline(0,color="black",lw=.8)
ax.set(xlabel="Noise level σ/H",ylabel="Distillation gain (pp)",title="A. Gain vs noise level",xticks=sigmas)
ax.legend(fontsize=8,ncol=2,frameon=False)

ax=axs[1]
x_aw=np.array([1.,1.5,2.])
for vals,label,color,marker in [(awgn_k,"AWGN K","#2368a0","o"),(awgn_r,"AWGN R","#c23b22","s")]:
    m=vals.mean(1); lo=vals.min(1); hi=vals.max(1)
    ax.plot(x_aw,m,marker+"-",color=color,lw=1.8,label=label)
    ax.vlines(x_aw,lo,hi,color=color,alpha=.55,lw=2)
# HBC at sigma 1.0; small jitter is only for readability.
for i,sc in enumerate(scenarios):
    x=1.0+(i-1.5)*.045
    for vals,color,marker in [(hbc_k[i],f"C{i+1}","^") ,(hbc_r[i],f"C{i+1}","v")]:
        ax.scatter([x,x],vals,marker=marker,color=color,s=36,alpha=.85)
        ax.plot([x,x],vals,color=color,alpha=.3,lw=1)
ax.scatter([1.5],[full15["K_pp"]],marker="^",color="C4",s=45,label="HBC full σ=1.5 K (seed 42 only)")
ax.scatter([1.5],[full15["R_pp"]],marker="v",color="C4",s=45,label="HBC full σ=1.5 R (seed 42 only)")
ax.axhline(0,color="black",lw=.8)
ax.set(xlabel="Noise level σ/H",ylabel="K or R contribution (pp)",title="B. K/R decomposition",xticks=[1.,1.5,2.])
ax.text(.02,.02,"HBC σ=1.0: colors by scenario; triangles=K, inverted triangles=R.\nHBC full σ=1.5 is single-seed, exploratory.",transform=ax.transAxes,fontsize=8,va="bottom")
ax.legend(fontsize=8,frameon=False,ncol=2)
fig.suptitle("Distillation robustness: AWGN and HBC noise summaries",fontsize=13)
fig.savefig(OUT,dpi=220,bbox_inches="tight")
print(OUT)
