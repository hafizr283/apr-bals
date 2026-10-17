#!/usr/bin/env python3
import os,time,subprocess,json,re
from pathlib import Path
from run_experiments import R,DATA,DATASETS,BASE,build,run
from gram_check import probe
# The first runner owns GPU1 until exit; do not overlap timings.
while Path('/proc/14817').exists():time.sleep(5)
# Select a spill-free contraction-loop unroll using timing only; preserve pilot logs.
choices=[]
for unroll in ('unroll1','unroll4','unroll8','outer'):
 log=run(f'sameflow_k96_{unroll}','ml20m','control_tuning_runs',extra={'CHECK_EVERY':'1','MAX_ITERS':'20','FIXED_ITERS':'1'})
 wall=float(re.findall(r'wall=([\d.]+) ms',log.read_text())[-1]);choices.append((wall,unroll))
best=min(choices)[1]
(R/'control_tuning_runs/selection.json').write_text(json.dumps({'criterion':'minimum ML20M wall time; accuracy not used; all candidates zero-spill','wall_ms_unroll':choices,'selected_unroll':best},indent=2))
p=R/'src/wmma_kernels.cuh';source=p.read_text()
if best=='outer':
 tuned=(R/'control_tuning_outer/wmma_kernels.cuh').read_text()
 a=tuned.index('                #pragma unroll 1\n                for (int kk')
 b=tuned.index('                ++tile;',a)
 x=source.index('                #pragma unroll\n                for (int e = 0; e < 8; ++e)')
 y=source.index('                ++tile;',x)
 source=source[:x]+'#if K_DIM == 96\n'+tuned[a:b]+'#else\n'+source[x:y]+'#endif\n'+source[y:]
else:
 source=source.replace('#pragma unroll 1\n#else',f'#pragma unroll {best[6:]}\n#else')
p.write_text(source)
import shutil
archive=R/'control_tuning_initial';archive.mkdir(exist_ok=True)
for p in (R/'four_way').glob('*sameflow_k96.*'):shutil.copy2(p,archive/p.name)
build('sameflow_k96',96,['-DWMMA_TF32=2','-DGIANT_NNZ_THRESH=2000000000','-DFAST_RMSE=0'])
for ds in DATASETS:run('sameflow_k96',ds,'four_way',extra={'CHECK_EVERY':'1','MAX_ITERS':'20','FIXED_ITERS':'1'})
# Rounded-input FP32 FMA control isolates operand quantization from atomics/MMA.
for k in (16,32,48,64,96):
 name=build(f'rounded_fma_k{k}',k,['-DWMMA_TF32=2','-DROUND_INPUT16=1','-DGIANT_NNZ_THRESH=2000000000','-DFAST_RMSE=0'])
 for ds in DATASETS:
  run(name,ds,'rounding_control',extra={'CHECK_EVERY':'1','MAX_ITERS':'20','FIXED_ITERS':'1'})
(R/'gram').mkdir(exist_ok=True)
for k in (16,32,48,64,96):
 cmd=BASE+[f'-DK_DIM={k}','-DWMMA_TF32=2',str(R/'src/gram_probe.cu'),'-o',str(R/'bin'/f'gram_probe_k{k}')]
 with open(R/'logs'/f'build_gram_probe_k{k}.txt','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 for ds,stem in DATASETS.items():
  prefix=R/'gram'/f'{ds}_k{k}'
  run(f'fp16_k{k}',ds,'gram',tag='_factors',extra={'CHECK_EVERY':'20','MAX_ITERS':'20','FIXED_ITERS':'1','DUMP_FACTORS':str(prefix)})
  probe(DATA/(stem+'.bin'),k,prefix,prefix)
  print('GRAM DONE',ds,k,flush=True)
for ds in DATASETS:
 for beta in ('0.3','0.5'):
  try:
   run('bf16_k96',ds,'bf16',tag='_beta'+beta,extra={'CHECK_EVERY':'1','MAX_ITERS':'30','FIXED_ITERS':'1','MBETA':beta})
  except RuntimeError as e:
   print('BF16 FAILED CONFIG',ds,beta,str(e),flush=True)
# Same-source cumulative ladder; record true gains even if non-monotone.
for name,flags in [('ladder_atomic',['-DATOMIC_BASELINE=1','-DDENSE_NNZ_THRESH=2000000000','-DFAST_RMSE=0']),('ladder_sameflow',['-DWMMA_TF32=2','-DGIANT_NNZ_THRESH=2000000000','-DFAST_RMSE=0']),('ladder_fp32',['-DCHOL_MP=0','-DFAST_RMSE=0']),('ladder_fp16solve',['-DCHOL_MP=2','-DFAST_RMSE=0']),('ladder_fastcheck',['-DCHOL_MP=2','-DFAST_RMSE=1']),('ladder_momentum',['-DCHOL_MP=2','-DFAST_RMSE=1','-DMOMENTUM=1'])]:
 build(name,96,flags)
 run(name,'netflix','cumulative',extra={'CHECK_EVERY':'1','MAX_ITERS':'30'})
# Access check and hardware stalls; failure retained instead of invented counters.
(R/'ncu').mkdir(exist_ok=True)
cmd=['/usr/local/cuda/bin/ncu','--target-processes','all','--kernel-name','regex:compute_LHS_RHS_wmma<6>','--launch-skip','8','--launch-count','1','--section','WarpStateStats','--section','ComputeWorkloadAnalysis','--csv',str(R/'bin/ship_k96'),str(DATA/'netflix_ratings.bin'),'.048']
with open(R/'ncu/stalls.txt','w') as f:subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',MAX_ITERS='2',FIXED_ITERS='1'),stdout=f,stderr=subprocess.STDOUT)
print('FOLLOWUPS DONE',flush=True)

(R/'logs/followups.done').write_text('completed\n')
