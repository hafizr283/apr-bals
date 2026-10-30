#!/usr/bin/env python3
import sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent))
from count_empty_entities import read_bin,census
from convert_bin_for_ccd_sgd import write_ccd,write_sgd
from make_validation_split import write_bin
for ds,stem in {'ml10m':'ratings10','ml20m':'ratings','ml32m':'ratings32','netflix':'netflix_ratings'}.items():
 d=read_bin(R/'splits'/(stem+'_validation.bin'));c=census(d);keep=~c['mask_any']
 for key in ('te_u','te_i','te_r'):d[key]=d[key][keep]
 d['nnz_test']=int(keep.sum())
 out=R/'external_inputs'/ds;out.mkdir(parents=True,exist_ok=True)
 write_bin(out/'warm_validation.bin',d,tuple(d[k] for k in ('tr_u','tr_i','tr_r')),tuple(d[k] for k in ('te_u','te_i','te_r')))
 write_ccd(d,out/'ccd');write_sgd(d,out/'sgd')
 (out/'manifest.json').write_text(json.dumps({'source':str(R/'splits'/(stem+'_validation.bin')),'scope':'validation ratings with both entities present in training; same mask for every method','removed_cold':int((~keep).sum()),'kept':int(keep.sum())},indent=2))
 print(ds,'done',flush=True)
