#!/usr/bin/env python3
import csv,json,struct
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
rows=[]
for p in sorted((R/'gram').glob('*_k*.json')):
 data=json.loads(p.read_text())
 if not isinstance(data,list):continue
 for side in ('users','items'):
  records=[r for r in data if r['side']==side and r['path']=='dense_fp16']
  if not records:continue
  ids=[r['entity'] for r in records];blocks=sorted(set(i//32 for i in ids))
  inp=R/'gram'/f'{p.stem}_{side}.input.bin'
  with open(inp,'rb') as f:
   n,nf,k,nnz=struct.unpack('<4i',f.read(16))
   off=np.fromfile(f,'<i4',n+1);idx=np.fromfile(f,'<i4',nnz);np.fromfile(f,'<f4',nnz)
   feat=np.fromfile(f,'<f4',nf*k).reshape(nf,k)
  original=np.fromfile(R/'gram'/f'{p.stem}_{side}.output.bin','<f4').reshape(4,n,k,k)
  cp=R/'accumulation_control'/f'{p.stem}_{side}.output.bin'
  chunked=np.fromfile(cp,'<f4').reshape(4,n,k,k) if cp.exists() else None
  upper=np.triu_indices(k)
  for r in records:
   ent=r['entity'];e=blocks.index(ent//32)*32+ent%32
   F=feat[idx[off[e]:off[e+1]]].astype(np.float64);ref=F.T@F
   H=F.astype(np.float16).astype(np.float64);h=H.T@H
   matrices=[(label,original[j,e]) for j,label in enumerate(('atomic_fp32','dense_fp16','sameflow_fp32','giant_fp16'))]
   if chunked is not None:matrices.append(('chunked_fp16',chunked[1,e]))
   for label,m in matrices:
    m=m.astype(np.float64)
    rows.append(dict(dataset=p.stem.split('_k')[0],K=k,side=side,entity=ent,nnz=r['nnz'],path=label,relative_error=np.linalg.norm((m-ref)[upper])/np.linalg.norm(ref[upper]),relative_error_to_rounded=np.linalg.norm((m-h)[upper])/np.linalg.norm(h[upper]),trace_relative_bias=np.trace(m-ref)/np.trace(ref),trace_bias_to_rounded=np.trace(m-h)/np.trace(h),gram_scale_bias=np.dot((m-ref)[upper],ref[upper])/np.dot(ref[upper],ref[upper])))
if rows:
 with open(R/'accumulation_errors.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(len(rows),'accumulation records')
for label in sorted(set(r['path'] for r in rows)):
 sub=[r for r in rows if r['path']==label]
 print(label,'median err',np.median([r['relative_error'] for r in sub]),'median trace bias',np.median([r['trace_relative_bias'] for r in sub]),'max err',max(r['relative_error'] for r in sub))
