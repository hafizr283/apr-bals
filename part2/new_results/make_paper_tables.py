#!/usr/bin/env python3
"""Generate reviewable LaTeX tables from completed CSV measurements only."""
import csv
from pathlib import Path
R=Path(__file__).resolve().parent
O=R/'paper_tables';O.mkdir(exist_ok=True)
def read(n):
 p=R/n
 return list(csv.DictReader(p.open())) if p.exists() else []
def table(name,columns,rows,caption):
 text=['% Generated from new_results CSV; check protocol notes before manuscript use.',r'\begin{table}',r'\centering',r'\caption{'+caption+'}',r'\begin{tabular}{'+'l'*len(columns)+'}',r'\hline',' & '.join(columns)+r' \\',r'\hline']
 text+=[' & '.join(map(str,row))+r' \\' for row in rows]
 text +=[r'\hline',r'\end{tabular}',r'\end{table}']
 (O/name).write_text('\n'.join(text)+'\n')
rows=read('four_way.csv');d={(r['dataset'],r['K'],r['variant']):r for r in rows};out=[]
for ds,k in sorted({(r['dataset'],r['K']) for r in rows},key=lambda x:(x[0],int(x[1]))):
 if all((ds,k,v) in d for v in ('atomic','sameflow','tf32','fp16')):
  out.append([ds,k]+[f"{float(d[ds,k,v]['wall_s']):.4f}" for v in ('atomic','sameflow','tf32','fp16')])
table('four_way.tex',['Dataset','$K$','Atomic','FP32 FMA','TF32','FP16'],out,'Training-loop seconds for 20 fixed sweeps; FP32 solver, no momentum or giant tier.')
val=read('validation_selection.csv')
table('validation.tex',['Dataset','$\\lambda$','Validation RMSE'],[[r['dataset'],r['selected_lambda'],r['validation_rmse']] for r in val],'Validation selection at rank 96 and sweep 30 on the new training split.')
rows=read('external_time_to_target.csv');out=[]
for r in rows:
 if r['K'] in ('96','128'):
  out.append([r['dataset'],r['method'],r['K'],r['target'],f"{float(r['seconds_to_target']):.4f}" if r['seconds_to_target'] else 'unreached',r['final_validation_rmse']])
table('external.tex',['Dataset','Method','Rank label','Target','Seconds','Final RMSE'],out,'Time to the common warm-validation target. cuMF-ALS uses rank 100 for the rank-96 pairing; SGD uses rank 128.')
print('Wrote',O)
rows=read('validation_accuracy_deltas.csv')
table('accuracy_pair.tex',['Dataset','$K$','FP32 RMSE','FP16 RMSE','Change (\\%)'],[[r['dataset'],r['K'],r['FP32_test'],r['FP16_test'],f"{float(r['relative_percent']):.5f}"] for r in rows],'Test accuracy after validation selection. Both paths use FP32 solvers and thirty fixed sweeps without momentum or the giant tier.')
rows=read('accumulation_errors.csv')
if rows:
 import statistics
 out=[]
 for path in ('atomic_fp32','sameflow_fp32','dense_fp16','giant_fp16','chunked_fp16'):
  rr=[r for r in rows if r['path']==path]
  if rr:out.append([path.replace('_',r'\_'),f"{statistics.median(float(r['relative_error']) for r in rr):.3g}",f"{max(float(r['relative_error']) for r in rr):.3g}",f"{statistics.median(float(r['trace_relative_bias']) for r in rr):.3g}"])
 table('gram_error.tex',['Path','Median error','Maximum error','Median trace bias'],out,'Relative upper-triangle Gram errors against FP64 on identical factors, with 800 heavy-entity snapshots per path.')
