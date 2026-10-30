import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent))
from count_empty_entities import read_bin
reports=[]
for stem in ('ratings10','ratings','ratings32','netflix_ratings'):
 original=read_bin('/home/pc/Desktop/2007080/'+stem+'.bin')
 v=read_bin(R/'splits'/(stem+'_validation.bin'));t=read_bin(R/'splits'/(stem+'_test.bin'))
 ix=np.load(R/'splits'/(stem+'_validation_indices.npy'));mask=np.ones(original['nnz_train'],bool);mask[ix]=False
 assert len(np.unique(ix))==len(ix)
 for key in ('tr_u','tr_i','tr_r'):
  assert np.array_equal(v[key],t[key])
  assert np.array_equal(original[key][mask],v[key])
 for tr,te in zip(('tr_u','tr_i','tr_r'),('te_u','te_i','te_r')):
  assert np.array_equal(original[tr][ix],v[te])
  assert np.array_equal(original[te],t[te])
 reports.append({'dataset':stem,'training_validation_disjoint_by_original_row':True,'training_plus_validation_exact_original_training':True,'test_arrays_bit_identical':True,'same_training_for_val_and_test':True})
 print(stem,'PASS',flush=True)
(R/'splits/verification.json').write_text(json.dumps(reports,indent=2))
