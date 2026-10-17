#!/usr/bin/env python3
"""Serial GPU runner; records exact command/environment and stops on failed runs."""
import subprocess,os,json,time
from pathlib import Path
R=Path(__file__).resolve().parent
DATA=Path('/home/pc/Desktop/2007080')
BASE=['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_86','-std=c++14','-DWEIGHTED_LAMBDA=1','-DCUMF_INIT=1','-Xptxas=-v']
DATASETS={'ml10m':'ratings10','ml20m':'ratings','ml32m':'ratings32','netflix':'netflix_ratings'}
def build(name,k,flags):
 cmd=BASE+[f'-DK_DIM={k}']+flags+[str(R/'src/justapr.cu'),'-o',str(R/'bin'/name)]
 with open(R/'logs'/('build_'+name+'.txt'),'w') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 return name
def run(name,dataset,group,tag='',lam=.048,extra=None,path=None):
 folder=R/group;folder.mkdir(exist_ok=True)
 log=folder/f'{dataset}_{name}{tag}.txt'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='1');env.update(extra or {})
 cmd=[str(R/'bin'/name),str(path or DATA/(DATASETS[dataset]+'.bin')),str(lam)]
 meta={'command':cmd,'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CHECK_EVERY','FIXED_ITERS','MAX_ITERS','MBETA','DUMP_FACTORS') if k in env},'start':time.time()}
 with open(log,'w') as f: subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 meta['end']=time.time();log.with_suffix('.json').write_text(json.dumps(meta,indent=2))
 text=log.read_text()
 if '=== APR-BALS final ===' not in text or 'nan' in text.lower() or 'error' in text.lower(): raise RuntimeError(log)
 print(log,flush=True)
 return log

def main():
 # Four-way controls keep FP32 solver, no momentum, no giant tier, exact RMSE.
 for k in (16,32,48,64,96):
  for variant,flags in [('fp16',[]),('tf32',['-DWMMA_TF32=1']),('sameflow',['-DWMMA_TF32=2']),('atomic',['-DDENSE_NNZ_THRESH=2000000000','-DATOMIC_BASELINE=1'])]:
   name=build(f'{variant}_k{k}',k,flags+['-DGIANT_NNZ_THRESH=2000000000','-DFAST_RMSE=0'])
   for ds in DATASETS:
    run(name,ds,'four_way',extra={'CHECK_EVERY':'1','MAX_ITERS':'20','FIXED_ITERS':'1'})
 # Per-sweep momentum, shipped solver and giant policy; exact RMSE for crossings.
 for k in (16,32,48,64,96):
  name=build(f'ship_k{k}',k,['-DMOMENTUM=1','-DFAST_RMSE=0']+(['-DCHOL_MP=2'] if k==96 else []))
  for ds in DATASETS:
   for beta in ('0','0.3'):
    run(name,ds,'momentum',tag='_beta'+beta,extra={'CHECK_EVERY':'1','MAX_ITERS':'30','FIXED_ITERS':'1','MBETA':beta})
 # Selection uses validation only. Choose per-dataset lambda using fixed 30 sweeps.
 for ds,stem in DATASETS.items():
  candidates=[]
  for lam in (.01,.02,.03,.04,.048,.06,.08,.12):
   log=run('ship_k96',ds,'validation',tag=f'_lambda{lam}',lam=lam,path=R/'splits'/(stem+'_validation.bin'),extra={'CHECK_EVERY':'1','MAX_ITERS':'30','FIXED_ITERS':'1','MBETA':'0.3'})
   import re
   rmse=float(re.findall(r'Train RMSE: [\d.]+ \| Test RMSE: ([\d.]+) \| compute=',log.read_text())[-1]);candidates.append((rmse,lam))
  best=min(candidates)
  (R/'validation'/f'{ds}_selection.json').write_text(json.dumps({'selection_rule':'minimum validation RMSE at sweep 30, K=96, beta=.3; tie smaller lambda','candidates':candidates,'selected_lambda':best[1]},indent=2))
  for k in (16,32,48,64,96):
   run(f'ship_k{k}',ds,'heldout_test',lam=best[1],path=R/'splits'/(stem+'_test.bin'),extra={'CHECK_EVERY':'30','MAX_ITERS':'30','FIXED_ITERS':'1','MBETA':'0.3'})
if __name__=='__main__': main()
