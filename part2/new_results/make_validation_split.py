#!/usr/bin/env python3
"""Carve 1/8 of original training ratings into validation; leave old test untouched.
Seed fixed before experiments. IDs preserved; sparse structures rebuilt from train only.
Validation .bin's 'test' field contains validation, never the original test.
"""
import sys,struct,json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from count_empty_entities import read_bin

def write_bin(path,d,train,held):
 u,i,r=train;v,j,s=held
 csr=coo_matrix((r,(u,i)),shape=(d['num_users'],d['num_items'])).tocsr()
 csc=csr.tocsc()
 # Duplicates cannot be silently merged: they would change the objective.
 if csr.nnz != len(r): raise ValueError('Duplicate user-item pairs require multiplicity-preserving CSR')
 with open(path,'wb') as f:
  f.write(struct.pack('<5i',d['version'],d['num_users'],d['num_items'],len(r),len(s)))
  for a in (u,i,r,v,j,s,csr.indptr,csr.indices,csr.data,csc.indptr,csc.indices,csc.data):
   a=np.asarray(a,dtype='<f4' if a.dtype.kind=='f' else '<i4')
   f.write(struct.pack('<Q',len(a)));a.tofile(f)

def main():
 for name in sys.argv[1:]:
  p=Path(name);d=read_bin(p);rng=np.random.default_rng(20261014)
  # Uniform random carve-out; keep >=1 training rating per nonempty user.
  valid=rng.random(d['nnz_train'])<0.125
  counts=np.bincount(d['tr_u'][~valid],minlength=d['num_users'])
  for u in np.flatnonzero(counts==0):
   ix=np.flatnonzero(d['tr_u']==u)
   if len(ix): valid[ix[0]]=False
  tr=tuple(d[k][~valid] for k in ('tr_u','tr_i','tr_r'))
  va=tuple(d[k][valid] for k in ('tr_u','tr_i','tr_r'))
  te=tuple(d[k] for k in ('te_u','te_i','te_r'))
  out=Path('new_results/splits');out.mkdir(exist_ok=True)
  write_bin(out/(p.stem+'_validation.bin'),d,tr,va)
  write_bin(out/(p.stem+'_test.bin'),d,tr,te)
  np.save(out/(p.stem+'_validation_indices.npy'),np.flatnonzero(valid))
  report={'source':str(p),'seed':20261014,'train':len(tr[0]),'validation':len(va[0]),'test':len(te[0]),'original_test_preserved':True,'selection':'validation only','test_caveat':'Original test was used in historical tuning; this carve-out cannot undo historical exposure. A fresh external test would be stronger.','validation_index_sha256':hashlib.sha256(np.flatnonzero(valid).tobytes()).hexdigest()}
  (out/(p.stem+'_manifest.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
