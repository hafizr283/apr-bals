#!/usr/bin/env python3
import csv,json,re
from pathlib import Path
R=Path(__file__).resolve().parent
p=R/'external_time_to_target.csv';rows=list(csv.DictReader(p.open())) if p.exists() else []
from summarize_external import curves
checks=[]
for (ds,method,k),curve in curves.items():
 expected=40 if method=='sgd' else 30
 checks.append({'kind':'sweep_sequence_and_monotonic_time','dataset':ds,'method':method,'K':k,'pass':[x[0] for x in curve]==list(range(1,expected+1)) and all(b[2]>a[2]>=0 for a,b in zip(curve,curve[1:]))})
for r in rows:
 expected=40 if r['method']=='sgd' else 30
 checks.append({'kind':'curve_length','dataset':r['dataset'],'method':r['method'],'K':r['K'],'pass':int(r['observed_sweeps'])==expected,'observed':int(r['observed_sweeps']),'expected':expected})
for ds in ('ml10m','ml20m','ml32m','netflix'):
 g=R/'external'/f'{ds}_sgd_k128.txt';c=R/'external'/f'{ds}_sgd_final_prediction.txt'
 if not (g.exists() and c.exists()):continue
 gm=re.findall(r'Validation RMSE: ([\d.eE+-]+)',g.read_text());cm=re.findall(r'RMSE = ([\d.eE+-]+)',c.read_text())
 if gm and cm:
  a,b=float(gm[-1]),float(cm[-1]);checks.append({'kind':'sgd_gpu_cpu_agreement','dataset':ds,'GPU':a,'CPU_printed':b,'tolerance':.000051,'pass':abs(a-b)<=.000051})
(R/'external_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(len(rows),'/84 curves;',sum(c['kind']=='sgd_gpu_cpu_agreement' for c in checks),'/4 SGD checks')
for c in checks:
 if not c['pass']:print('FAIL',c)
