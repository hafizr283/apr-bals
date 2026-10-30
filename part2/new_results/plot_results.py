#!/usr/bin/env python3
"""Figures from new logs only; skip incomplete panels rather than invent cells."""
from pathlib import Path
import csv,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize import R,curves
O=R/'figures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
def save(fig,name):
 fig.tight_layout();fig.savefig(O/(name+'.pdf'));fig.savefig(O/(name+'.png'),dpi=180);plt.close(fig)
if (R/'four_way.csv').exists():
 rows=list(csv.DictReader(open(R/'four_way.csv')));fig,axs=plt.subplots(1,4,figsize=(12,3),sharey=True)
 for ax,ds in zip(axs,('ml10m','ml20m','ml32m','netflix')):
  for label in ('atomic','sameflow','tf32','fp16'):
   rr=sorted([r for r in rows if r['dataset']==ds and r['variant']==label],key=lambda r:int(r['K']))
   ax.plot([int(r['K']) for r in rr],[float(r['wall_s'])/20 for r in rr],'-o',label=label,ms=3)
  ax.set(title=ds,xlabel='Rank K',yscale='log');ax.grid(alpha=.2)
 axs[0].set_ylabel('Seconds per sweep (incl. evaluation)');axs[-1].legend(fontsize=8);save(fig,'four_way')
fig,axs=plt.subplots(1,4,figsize=(12,3))
for ax,ds in zip(axs,('ml10m','ml20m','ml32m','netflix')):
 for beta in ('0','0.3'):
  p=R/'momentum'/f'{ds}_ship_k96_beta{beta}.txt'
  if p.exists():
   c=curves(p);ax.plot([x[0] for x in c],[x[2] for x in c],label='beta='+beta)
 ax.set(title=ds,xlabel='Sweep',ylabel='Held-out RMSE');ax.grid(alpha=.2)
 if ax.lines:ax.legend()
save(fig,'momentum_k96')
fig,axs=plt.subplots(1,4,figsize=(12,3))
for ax,ds in zip(axs,('ml10m','ml20m','ml32m','netflix')):
 p=R/'validation'/f'{ds}_selection.json'
 if p.exists():
  v=json.loads(p.read_text());c=sorted(v['candidates'],key=lambda x:x[1]);ax.plot([x[1] for x in c],[x[0] for x in c],'-o',ms=3);ax.axvline(v['selected_lambda'],color='k',ls=':',lw=.8)
 ax.set(title=ds,xlabel='Lambda',ylabel='Validation RMSE');ax.grid(alpha=.2)
save(fig,'validation_lambda')
if (R/'gram_errors.csv').exists():
 rows=list(csv.DictReader(open(R/'gram_errors.csv')));labels=['atomic_fp32','sameflow_fp32','dense_fp16','giant_fp16'];fig,ax=plt.subplots(figsize=(6,3.5))
 vals=[[max(float(r['relative_upper_frobenius']),1e-16) for r in rows if r['path']==label] for label in labels]
 ax.boxplot(vals,labels=['Atomic FP32','Dataflow FP32','Dense FP16','Giant FP16']);ax.set(yscale='log',ylabel='Relative upper-triangle Frobenius error');ax.grid(axis='y',alpha=.2);save(fig,'gram_error')
fig,axs=plt.subplots(5,4,figsize=(12,12),sharex=True)
for row,k in enumerate((16,32,48,64,96)):
 for col,ds in enumerate(('ml10m','ml20m','ml32m','netflix')):
  ax=axs[row,col]
  for beta in ('0','0.3'):
   p=R/'momentum'/f'{ds}_ship_k{k}_beta{beta}.txt'
   if p.exists():
    c=curves(p);ax.plot([x[0] for x in c],[x[2] for x in c],label='beta='+beta)
  ax.set(title=f'{ds}, K={k}');ax.grid(alpha=.2)
  if row==4:ax.set_xlabel('Sweep')
  if col==0:ax.set_ylabel('Held-out RMSE')
axs[0,-1].legend();save(fig,'momentum_all_ranks')
if (R/'accumulation_errors.csv').exists():
 rows=list(csv.DictReader(open(R/'accumulation_errors.csv')));fig,axs=plt.subplots(1,2,figsize=(10,3.7))
 for label in ('atomic_fp32','sameflow_fp32','dense_fp16','giant_fp16','chunked_fp16'):
  rr=[r for r in rows if r['path']==label]
  if not rr:continue
  axs[0].scatter([int(r['nnz']) for r in rr],[float(r['relative_error']) for r in rr],s=6,alpha=.3,label=label)
  axs[1].scatter([int(r['nnz']) for r in rr],[float(r['trace_relative_bias']) for r in rr],s=6,alpha=.3,label=label)
 axs[0].set(xscale='log',yscale='log',xlabel='Entity training observations',ylabel='Relative Gram error')
 axs[1].set(xscale='log',xlabel='Entity training observations',ylabel='Relative trace bias')
 axs[1].axhline(0,color='k',lw=.5);axs[0].legend(fontsize=7)
 for ax in axs:ax.grid(alpha=.2)
 save(fig,'accumulation_bias')
