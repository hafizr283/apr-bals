#!/usr/bin/env python3
import re,json,csv
from pathlib import Path
R=Path(__file__).resolve().parent

def curves(path):
 s=path.read_text(encoding="utf-8")
 return [(int(a),float(b),float(c),float(d)/1000) for a,b,c,d in re.findall(r'Iter\s+(\d+) \| Train RMSE: ([\d.eE+-]+) \| Test RMSE: ([\d.eE+-]+)\s+CUMULATIVE_WALL_MS ([\d.eE+-]+)',s)]
def final(path):
 s=path.read_text(encoding="utf-8");m=re.findall(r'Train RMSE: ([\d.eE+-]+) \| Test RMSE: ([\d.eE+-]+) \| compute=([\d.]+) ms \| wall=([\d.]+) ms',s)
 if not m:return None
 tr,te,comp,wall=map(float,m[-1]);gram=sum(float(v) for v in re.findall(r'(?:User|Item) LHS\+RHS:\s*([\d.]+) ms total',s));fail=re.findall(r'non-finite solves zeroed = (\d+)',s)
 return dict(train=tr,test=te,compute_s=comp/1000,wall_s=wall/1000,gram_s=gram/1000,zeroed_solves=int(fail[-1]) if fail else 0)
def csvout(name,rows):
 if not rows:return
 with open(R/name,'w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[];four={}
for p in sorted((R/'four_way').glob('*.txt')):
 m=re.match(r'(.*)_(fp16|tf32|sameflow|atomic)_k(\d+)',p.stem);r=final(p)
 if m and r:
  ds,v,k=m.groups();row=dict(dataset=ds,K=int(k),variant=v,**r);rows.append(row);four[ds,int(k),v]=r
csvout('four_way.csv',rows)
mom=[]
for ds in ('ml10m','ml20m','ml32m','netflix'):
 for k in (16,32,48,64,96):
  pair=[]
  for beta in ('0','0.3'):
   p=R/'momentum'/f'{ds}_ship_k{k}_beta{beta}.txt'
   if not p.exists():break
   c=curves(p)
   if len(c)<30:break
   # Report true one-sweep stopping and five-sweep-window stopping separately.
   hits=[r for prev,r in zip(c,c[1:]) if abs(prev[1]-r[1])<.001]
   window=[r for prev,r in zip(c,c[5:]) if abs(prev[1]-r[1])<.001]
   h=hits[0] if hits else (None,None,None,None);w=window[0] if window else (None,None,None,None)
   pair.append(dict(dataset=ds,K=k,beta=beta,first_delta1_sweep=h[0],first_delta1_wall_s=h[3],first_delta1_test=h[2],first_delta5_sweep=w[0],first_delta5_wall_s=w[3]))
  if len(pair)==2:mom.extend(pair)
csvout('momentum_crossings.csv',mom)
val=[]
for p in sorted((R/'validation').glob('*selection.json')):
 d=json.loads(p.read_text(encoding="utf-8"));val.append({'dataset':p.stem.replace('_selection',''),'selected_lambda':d['selected_lambda'],'validation_rmse':min(x[0] for x in d['candidates'])})
csvout('validation_selection.csv',val)
sweep=[]
for p in sorted((R/'validation').glob('*.txt')):
 r=final(p)
 if r:sweep.append(dict(run=p.stem,**r))
csvout('validation_sweep.csv',sweep)

gram=[]
for p in (R/'gram').glob('*_k*.json'):
 d=json.loads(p.read_text(encoding="utf-8"))
 if not isinstance(d,list):continue
 ds=p.stem.split('_k')[0]
 gram.extend(dict(dataset=ds,**r) for r in d)
csvout('gram_errors.csv',gram)
bf=[]
for p in (R/'bf16').glob('*.txt'):
 r=final(p)
 if r:bf.append(dict(run=p.stem,**r))
csvout('bf16.csv',bf)
bfc=[]
for p in sorted((R/'bf16_controls').glob('*.txt')):
 r=final(p)
 if r:bfc.append(dict(run=p.stem,**r))
csvout('bf16_controls.csv',bfc)
bflookup={r['run']:r for r in bfc};bfmatched=[]
for r in bf:
 ds,beta=r['run'].split('_bf16_k96_beta');f=bflookup.get(f'{ds}_fp16_ir1_k96_beta{beta}')
 if f:bfmatched.append(dict(dataset=ds,beta=beta,BF16_test=r['test'],FP16_refined_test=f['test'],BF16_minus_FP16=r['test']-f['test'],BF16_zeroed=r['zeroed_solves'],FP16_zeroed=f['zeroed_solves']))
csvout('bf16_matched_comparison.csv',bfmatched)


held=[]
for p in sorted((R/'heldout_test').glob('*.txt')):
 r=final(p)
 if r:held.append(dict(run=p.stem,**r))
csvout('heldout_test.csv',held)
accuracy=[]
for p in sorted((R/'validation_accuracy_pair').glob('*.txt')):
 r=final(p)
 if r:accuracy.append(dict(run=p.stem,**r))
csvout('validation_accuracy_pair.csv',accuracy)
pairs={}
for r in accuracy:
 m=re.match(r'(.*)_(fp16|sameflow)_k(\d+)',r['run'])
 if m:
  ds,v,k=m.groups();pairs.setdefault((ds,int(k)),{})[v]=r
accuracy_deltas=[]
for (ds,k),pair in sorted(pairs.items()):
 if len(pair)==2:
  a,b=pair['sameflow']['test'],pair['fp16']['test']
  accuracy_deltas.append(dict(dataset=ds,K=k,FP32_test=a,FP16_test=b,FP16_minus_FP32=b-a,relative_percent=100*(b-a)/a))
csvout('validation_accuracy_deltas.csv',accuracy_deltas)


ladder=[]
for p in sorted((R/'cumulative').glob('*.txt')):
 r=final(p)
 if r:ladder.append(dict(run=p.stem,sweeps=len(curves(p)),**r))
csvout('cumulative.csv',ladder)
rounded=[]
for p in sorted((R/'rounding_control').glob('*.txt')):
 m=re.match(r'(.*)_rounded_fma_k(\d+)',p.stem);r=final(p)
 if m and r:
  ds,k=m.groups();k=int(k);f=four.get((ds,k,'fp16'));a=four.get((ds,k,'sameflow'))
  if f and a:rounded.append(dict(dataset=ds,K=k,rounded_fma_test=r['test'],fp16_test=f['test'],sameflow_test=a['test'],rounded_minus_fp16=r['test']-f['test'],rounded_minus_sameflow=r['test']-a['test']))
csvout('rounding_control.csv',rounded)
chunked=[]
for p in sorted((R/'accumulation_control').glob('*chunked_fp16*.txt')):
 m=re.match(r'(.*)_chunked_fp16_k(\d+)',p.stem);r=final(p)
 if m and r:
  ds,k=m.groups();k=int(k);f=four.get((ds,k,'fp16'));a=four.get((ds,k,'sameflow'))
  if f and a:chunked.append(dict(dataset=ds,K=k,chunked_test=r['test'],fp16_test=f['test'],sameflow_test=a['test'],chunked_train=r['train'],fp16_train=f['train'],sameflow_train=a['train'],chunked_minus_fp16=r['test']-f['test'],chunked_minus_sameflow=r['test']-a['test']))
csvout('accumulation_control.csv',chunked)

print("CSV summaries updated. Protocol and interpretation: docs/evidence.md and docs/reproduction.md")
