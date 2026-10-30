import json,re,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
out=[]
for k in (16,32,48,64,96):
 s=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(R/'bin'/f'sameflow_k{k}')],text=True)
 fn=f'_Z25compute_LHS_RHS_wmma_tf32ILi{k//16}EEviiPKiS1_PKfS3_PfS4_i'
 parts=s.split('Function : ')
 code=next(p for p in parts if p.splitlines()[0].strip()==fn)
 row={'K':k,'function':fn,'FFMA_count_static':len(re.findall(r'\bFFMA\b',code)),'HMMA_count_static':len(re.findall(r'\bHMMA',code))}
 assert row['FFMA_count_static'] and not row['HMMA_count_static'],row
 out.append(row)
(R/'sameflow_sass_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
