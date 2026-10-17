#!/usr/bin/env python3
import csv,re,json
from pathlib import Path
R=Path(__file__).resolve().parent
T=json.loads((R/'external_targets.json').read_text())['targets']
rows=[];curves={}
for p in (R/'external').glob('*.txt'):
 m=re.match(r'(ml10m|ml20m|ml32m|netflix)_(apr|als|ccd|bals|sgd)_k(\d+)',p.stem)
 if not m:continue
 ds,method,k=m.groups();k=int(k);s=p.read_text()
 if method=='ccd':
  c=[(int(i),float(r),float(t)/1000) for i,r,t in re.findall(r'iter (\d+) time [\d.]+ RMSE ([\d.eE+-]+)\s+CUMULATIVE_WALL_MS ([\d.eE+-]+)',s)]
 elif method=='sgd':
  c=[(int(i),float(r),float(t)/1000) for i,r,t in re.findall(r'Iter (\d+) \| Validation RMSE: ([\d.eE+-]+)\s+CUMULATIVE_WALL_MS ([\d.eE+-]+)',s)]
 else:
  c=[(int(i),float(r),float(t)/1000) for i,r,t in re.findall(r'Iter\s+(\d+) \| Train RMSE: [\d.eE+-]+ \| Test RMSE: ([\d.eE+-]+)\s+CUMULATIVE_WALL_MS ([\d.eE+-]+)',s)]
 if not c:continue
 curves[ds,method,k]=c
 hits=[x for x in c if x[1]<=T[ds]]
 rows.append(dict(dataset=ds,method=method,K=k,target=T[ds],first_sweep=hits[0][0] if hits else '',seconds_to_target=hits[0][2] if hits else '',final_validation_rmse=c[-1][1],observed_sweeps=len(c),reached=bool(hits)))
if rows:
 with open(R/'external_time_to_target.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
lines=['# External common-target comparison','', 'All curves use the same warm-validation ratings per dataset. Targets were saved in `external_targets.json` before these runs. Timings include periodic evaluation, exclude preprocessing/loading, and use the implementations\' instrumented training loops. Rank mismatch remains explicit. New source changes are documented in `EXTERNAL_PROTOCOL.md`.','', '| Dataset | Method | APR comparison rank | Target | First sweep | Seconds | Final validation RMSE |','|---|---|---:|---:|---:|---:|---:|']
for r in sorted(rows,key=lambda r:(r['dataset'],r['K'],r['method'])):
 lines.append(f"| {r['dataset']} | {r['method']} | {r['K']} | {r['target']} | {r['first_sweep'] or 'unreached'} | {str(round(r['seconds_to_target'],4)) if r['reached'] else 'censored'} | {r['final_validation_rmse']:.6f} |")
(R/'EXTERNAL_RESULTS.md').write_text('\n'.join(lines)+'\n');print(len(rows),'external curves')
headline=[]
for ds in T:
 selected={r['method']:r for r in rows if r['dataset']==ds and ((r['method']=='sgd' and r['K']==128) or (r['method']!='sgd' and r['K']==96))}
 if len(selected)!=5:continue
 a=selected['apr']
 for method,r in selected.items():
  headline.append(dict(dataset=ds,method=method,rank=100 if method=='als' else r['K'],target=T[ds],seconds_to_target=r['seconds_to_target'],ratio_to_apr=r['seconds_to_target']/a['seconds_to_target'] if r['reached'] and a['reached'] else '',final_validation_rmse=r['final_validation_rmse']))
if headline:
 with open(R/'external_k96_comparison.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=headline[0]);w.writeheader();w.writerows(headline)
