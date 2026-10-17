#!/usr/bin/env python3
"""Common warm-validation curves; no test-driven selection.
SGD uses one model with per-epoch GPU validation; saved-model CPU check verifies scoring.
"""
import os,subprocess,json,time,re
from pathlib import Path
from run_experiments import R,DATASETS,BASE
while not (R/'logs/followups.done').exists():time.sleep(5)
E=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MAX_ITERS='30',FIXED_ITERS='1',CHECK_EVERY='1',MBETA='0.3')
OUT=R/'external';OUT.mkdir(exist_ok=True)
def execute(cmd,log,env=E):
 start=time.time()
 with open(log,'w') as f:r=subprocess.run(list(map(str,cmd)),env=env,stdout=f,stderr=subprocess.STDOUT,cwd=OUT)
 log.with_suffix('.json').write_text(json.dumps({'command':list(map(str,cmd)),'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MAX_ITERS','FIXED_ITERS','CHECK_EVERY','MBETA')},'returncode':r.returncode,'start':start,'end':time.time()},indent=2))
 if r.returncode:raise RuntimeError(log)
 print(log,flush=True)
for k in (16,32,48,64,96):
 cmd=BASE+[f'-DK_DIM={k}','-DBALS_REORDER=1','-DFAST_RMSE=0',str(R/'src/bals.cu'),'-o',str(R/'bin'/f'bals_k{k}')]
 execute(cmd,R/'logs'/f'build_bals_k{k}.txt')
for ds,stem in DATASETS.items():
 inp=R/'external_inputs'/ds
 lam=json.loads((R/'validation'/f'{ds}_selection.json').read_text())['selected_lambda']
 # External CCD has its own regularizer. Select it on validation at K96.
 cands=[]
 for cl in (.02,.04,.05,.058,.08,.12):
  log=OUT/f'{ds}_ccd_select_{cl}.txt'
  execute([R/'bin/ccd','-T','1','-a','100000','-b','100000','-l',cl,'-k','96','-t','30',inp/'ccd'],log)
  matches=re.findall(r'iter\s+(\d+) time ([\d.]+) RMSE ([\d.eE+-]+)',log.read_text())
  cands.append((float(matches[-1][2]),cl))
 cl=min(cands)[1];(OUT/f'{ds}_ccd_selection.json').write_text(json.dumps({'selected_lambda':cl,'candidates':cands},indent=2))
 for k,f in zip((16,32,48,64,96),(20,30,50,60,100)):
  execute([R/'bin'/f'ship_k{k}',inp/'warm_validation.bin',lam],OUT/f'{ds}_apr_k{k}.txt')
  execute([R/'bin/cumf_als',inp/'warm_validation.bin',lam,f,'1','3','0','30','0'],OUT/f'{ds}_als_k{k}_f{f}.txt')
  execute([R/'bin/ccd','-T','1','-a','100000','-b','100000','-l',cl,'-k',k,'-t','30',inp/'ccd'],OUT/f'{ds}_ccd_k{k}.txt')
  execute([R/'bin'/f'bals_k{k}',inp/'warm_validation.bin',lam],OUT/f'{ds}_bals_k{k}.txt')
 # Prespecified published SGD settings, one continuous model with per-epoch checks.
 workers={'ml10m':256,'ml20m':384,'ml32m':512,'netflix':512}[ds]
 model=OUT/f'{ds}_sgd.model'
 execute([R/'bin/sgd','-g','0','-l','.05','-a','.08','-b','.3','-s',workers,'-k','128','-t','40',inp/'sgd/train.bin',model],OUT/f'{ds}_sgd_k128.txt',env=dict(E,REVIEW_VALIDATION=str(inp/'sgd/test.bin')))
 execute([R.parent/'cumf_sgd-master/test/mf-predict','-e','0',inp/'sgd/test.bin',model,'/dev/null'],OUT/f'{ds}_sgd_final_prediction.txt')
print('EXTERNAL DONE',flush=True)

(R/'logs/external.done').write_text('completed\n')
