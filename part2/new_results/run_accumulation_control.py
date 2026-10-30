#!/usr/bin/env python3
"""Isolate persistent MMA accumulation: reset MMA accumulator each 16-row chunk,
then combine chunk results using FP32 CUDA-core adds. No timing headline uses it."""
import os,time,subprocess,json
from pathlib import Path
from run_experiments import R,BASE,DATASETS,run
while not (R/'logs/followups.done').exists():time.sleep(5)
(R/'accumulation_control').mkdir(exist_ok=True)
for k in (16,32,48,64,96):
 name=f'chunked_fp16_k{k}'
 cmd=BASE+[f'-DK_DIM={k}','-DGIANT_NNZ_THRESH=2000000000','-DFAST_RMSE=0',str(R/'chunked_src/justapr.cu'),'-o',str(R/'bin'/name)]
 with open(R/'logs'/f'build_{name}.txt','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 for ds in DATASETS:run(name,ds,'accumulation_control',extra={'CUDA_VISIBLE_DEVICES':'0','CHECK_EVERY':'1','MAX_ITERS':'20','FIXED_ITERS':'1'})
 probe=R/'bin'/f'chunked_probe_k{k}'
 cmd=BASE+[f'-DK_DIM={k}','-DWMMA_TF32=2',str(R/'chunked_src/gram_probe.cu'),'-o',str(probe)]
 with open(R/'logs'/f'build_chunked_probe_k{k}.txt','w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 for ds in DATASETS:
  for side in ('users','items'):
   prefix=f'{ds}_k{k}_{side}'
   with open(R/'accumulation_control'/f'{prefix}.log','w') as f:
    subprocess.run([str(probe),str(R/'gram'/f'{prefix}.input.bin'),str(R/'accumulation_control'/f'{prefix}.output.bin')],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=f,stderr=subprocess.STDOUT,check=True)
# Controlled accuracy pair on the validation-selected split and lambda.
# Same FP32 solver, no momentum, no giant tier; test checked only after sweep 30.
for ds,stem in DATASETS.items():
 lam=json.loads((R/'validation'/f'{ds}_selection.json').read_text())['selected_lambda']
 for k in (16,32,48,64,96):
  for variant in ('sameflow','fp16'):
   run(f'{variant}_k{k}',ds,'validation_accuracy_pair',lam=lam,path=R/'splits'/f'{stem}_test.bin',extra={'CUDA_VISIBLE_DEVICES':'0','CHECK_EVERY':'30','MAX_ITERS':'30','FIXED_ITERS':'1'})
print('ACCUMULATION CONTROL DONE',flush=True)
(R/'logs/accumulation.done').write_text('completed\n')
