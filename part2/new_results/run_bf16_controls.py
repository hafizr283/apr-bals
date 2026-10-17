#!/usr/bin/env python3
"""GPU0 safety/accuracy controls; no cross-GPU performance claims."""
from run_experiments import R,build,run,DATASETS
build('fp16_ir1_k96',96,['-DMOMENTUM=1','-DCHOL_MP=2','-DCHOL_REFINE=1','-DFAST_RMSE=0'])
for ds in DATASETS:
 for name,beta in [('fp16_ir1_k96','0.3'),('fp16_ir1_k96','0.5'),('ship_k96','0.5')]:
  try:
   run(name,ds,'bf16_controls',tag='_beta'+beta,extra={'CUDA_VISIBLE_DEVICES':'0','CHECK_EVERY':'1','MAX_ITERS':'30','FIXED_ITERS':'1','MBETA':beta})
  except RuntimeError as e:print('FAILED CONFIG',str(e),flush=True)
(R/'logs/bf16_controls.done').write_text('completed\n')
