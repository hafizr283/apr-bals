#!/usr/bin/env python3
"""Actual CUDA Gram kernels vs NumPy FP64 on identical trained FP32 factors.
Upper triangles only (the contract consumed by Cholesky); no regularization.
"""
import os,sys,struct,json,subprocess
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent

def read_sparse(path):
 with open(path,'rb') as f:
  h=struct.unpack('<5i',f.read(20))
  def v(dtype):
   n=struct.unpack('<Q',f.read(8))[0];return np.fromfile(f,dtype=dtype,count=n)
  for _ in range(6):
   n=struct.unpack('<Q',f.read(8))[0];f.seek(n*4,1)
  return h, (v('<i4'),v('<i4'),v('<f4')), (v('<i4'),v('<i4'),v('<f4'))

def probe(dataset,k,factor_prefix,outprefix):
 h,users,items=read_sparse(dataset);_,nu,ni,_,_=h
 X=np.fromfile(str(factor_prefix)+'_X.bin',dtype='<f4').reshape(nu,k)
 Y=np.fromfile(str(factor_prefix)+'_Y.bin',dtype='<f4').reshape(ni,k)
 rows=[]
 for side,(off,idx,val),feat in [('users',users,Y),('items',items,X)]:
  ids=np.argsort(-np.diff(off),kind='stable')[:20]
  # Preserve every selected entity's original 32-row tile neighborhood.
  # Atomic job counts depend on all rows in that tile, not only the target row.
  blocks=sorted(set(int(ent)//32 for ent in ids))
  packed_ids=[ent for block in blocks for ent in range(block*32,min((block+1)*32,len(off)-1))]
  lookup={ent:e for e,ent in enumerate(packed_ids)}
  offsets=[0];indices=[];values=[]
  for ent in packed_ids:
   indices.append(idx[off[ent]:off[ent+1]]);values.append(val[off[ent]:off[ent+1]]);offsets.append(offsets[-1]+len(indices[-1]))
  indices=np.concatenate(indices);values=np.concatenate(values)
  input_path=Path(str(outprefix)+'_'+side+'.input.bin');output_path=Path(str(outprefix)+'_'+side+'.output.bin')
  with open(input_path,'wb') as f:
   f.write(struct.pack('<4i',len(packed_ids),len(feat),k,len(indices)))
   for a in (np.array(offsets,dtype='<i4'),indices,values,feat):a.tofile(f)
  result=subprocess.run([str(R/'bin'/f'gram_probe_k{k}'),str(input_path),str(output_path)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'),capture_output=True,text=True,check=True)
  Path(str(outprefix)+'_'+side+'.log').write_text(result.stdout+result.stderr)
  matrices=np.fromfile(output_path,dtype='<f4').reshape(4,len(packed_ids),k,k)
  upper=np.triu_indices(k)
  for ent in ids:
   e=lookup[int(ent)]
   factors=feat[indices[offsets[e]:offsets[e+1]]].astype(np.float64)
   ref=factors.T@factors;ref_upper=ref[upper];den=np.linalg.norm(ref_upper)
   rounded=factors.astype(np.float16).astype(np.float64);roundref=rounded.T@rounded
   for mode,label in enumerate(('atomic_fp32','dense_fp16','sameflow_fp32','giant_fp16')):
    observed=matrices[mode,e].astype(np.float64)[upper];err=observed-ref_upper
    rows.append({'side':side,'entity':int(ent),'nnz':len(factors),'K':k,'path':label,'relative_upper_frobenius':float(np.linalg.norm(err)/den),'max_abs_over_max_ref':float(np.max(np.abs(err))/np.max(np.abs(ref_upper))),'fp16_operand_error':float(np.linalg.norm(roundref[upper]-ref_upper)/den)})
  # Inputs/outputs retained for independent reproduction.
 Path(str(outprefix)+'.json').write_text(json.dumps(rows,indent=2));return rows
if __name__=='__main__':probe(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4])
