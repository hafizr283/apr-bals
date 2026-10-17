#!/usr/bin/env python3
import re,json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent

def curves(path):
 s=path.read_text()
 return [(int(a),float(b),float(c),float(d)/1000) for a,b,c,d in re.findall(r'Iter\s+(\d+) \| Train RMSE: ([\d.eE+-]+) \| Test RMSE: ([\d.eE+-]+)\s+CUMULATIVE_WALL_MS ([\d.eE+-]+)',s)]
def final(path):
 s=path.read_text();m=re.findall(r'Train RMSE: ([\d.eE+-]+) \| Test RMSE: ([\d.eE+-]+) \| compute=([\d.]+) ms \| wall=([\d.]+) ms',s)
 if not m:return None
 tr,te,comp,wall=map(float,m[-1]);gram=sum(float(v) for v in re.findall(r'(?:User|Item) LHS\+RHS:\s*([\d.]+) ms total',s));fail=re.findall(r'non-finite solves zeroed = (\d+)',s)
 return dict(train=tr,test=te,compute_s=comp/1000,wall_s=wall/1000,gram_s=gram/1000,zeroed_solves=int(fail[-1]) if fail else 0)
def csvout(name,rows):
 if not rows:return
 with open(R/name,'w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[];four={}
for p in sorted((R/'four_way').glob('*.txt')):
 m=re.match(r'(.*)_(fp16|tf32|sameflow|atomic)_k(\d+)',p.stem);r=final(p)
 if m and r:
  ds,v,k=m.groups();row=dict(dataset=ds,K=int(k),variant=v,**r);rows.append(row);four[ds,int(k),v]=r
csvout('four_way.csv',rows)
mom=[]
for ds in ('ml10m','ml20m','ml32m','netflix'):
 for k in (16,32,48,64,96):
  pair=[]
  for beta in ('0','0.3'):
   p=R/'momentum'/f'{ds}_ship_k{k}_beta{beta}.txt'
   if not p.exists():break
   c=curves(p)
   if len(c)<30:break
   # Report true one-sweep stopping and five-sweep-window stopping separately.
   hits=[r for prev,r in zip(c,c[1:]) if abs(prev[1]-r[1])<.001]
   window=[r for prev,r in zip(c,c[5:]) if abs(prev[1]-r[1])<.001]
   h=hits[0] if hits else (None,None,None,None);w=window[0] if window else (None,None,None,None)
   pair.append(dict(dataset=ds,K=k,beta=beta,first_delta1_sweep=h[0],first_delta1_wall_s=h[3],first_delta1_test=h[2],first_delta5_sweep=w[0],first_delta5_wall_s=w[3]))
  if len(pair)==2:mom.extend(pair)
csvout('momentum_crossings.csv',mom)
val=[]
for p in sorted((R/'validation').glob('*selection.json')):
 d=json.loads(p.read_text());val.append({'dataset':p.stem.replace('_selection',''),'selected_lambda':d['selected_lambda'],'validation_rmse':min(x[0] for x in d['candidates'])})
csvout('validation_selection.csv',val)
sweep=[]
for p in sorted((R/'validation').glob('*.txt')):
 r=final(p)
 if r:sweep.append(dict(run=p.stem,**r))
csvout('validation_sweep.csv',sweep)

gram=[]
for p in (R/'gram').glob('*_k*.json'):
 d=json.loads(p.read_text())
 if not isinstance(d,list):continue
 ds=p.stem.split('_k')[0]
 gram.extend(dict(dataset=ds,**r) for r in d)
csvout('gram_errors.csv',gram)
bf=[]
for p in (R/'bf16').glob('*.txt'):
 r=final(p)
 if r:bf.append(dict(run=p.stem,**r))
csvout('bf16.csv',bf)
bfc=[]
for p in sorted((R/'bf16_controls').glob('*.txt')):
 r=final(p)
 if r:bfc.append(dict(run=p.stem,**r))
csvout('bf16_controls.csv',bfc)
bflookup={r['run']:r for r in bfc};bfmatched=[]
for r in bf:
 ds,beta=r['run'].split('_bf16_k96_beta');f=bflookup.get(f'{ds}_fp16_ir1_k96_beta{beta}')
 if f:bfmatched.append(dict(dataset=ds,beta=beta,BF16_test=r['test'],FP16_refined_test=f['test'],BF16_minus_FP16=r['test']-f['test'],BF16_zeroed=r['zeroed_solves'],FP16_zeroed=f['zeroed_solves']))
csvout('bf16_matched_comparison.csv',bfmatched)


held=[]
for p in sorted((R/'heldout_test').glob('*.txt')):
 r=final(p)
 if r:held.append(dict(run=p.stem,**r))
csvout('heldout_test.csv',held)
accuracy=[]
for p in sorted((R/'validation_accuracy_pair').glob('*.txt')):
 r=final(p)
 if r:accuracy.append(dict(run=p.stem,**r))
csvout('validation_accuracy_pair.csv',accuracy)
pairs={}
for r in accuracy:
 m=re.match(r'(.*)_(fp16|sameflow)_k(\d+)',r['run'])
 if m:
  ds,v,k=m.groups();pairs.setdefault((ds,int(k)),{})[v]=r
accuracy_deltas=[]
for (ds,k),pair in sorted(pairs.items()):
 if len(pair)==2:
  a,b=pair['sameflow']['test'],pair['fp16']['test']
  accuracy_deltas.append(dict(dataset=ds,K=k,FP32_test=a,FP16_test=b,FP16_minus_FP32=b-a,relative_percent=100*(b-a)/a))
csvout('validation_accuracy_deltas.csv',accuracy_deltas)


ladder=[]
for p in sorted((R/'cumulative').glob('*.txt')):
 r=final(p)
 if r:ladder.append(dict(run=p.stem,sweeps=len(curves(p)),**r))
csvout('cumulative.csv',ladder)
rounded=[]
for p in sorted((R/'rounding_control').glob('*.txt')):
 m=re.match(r'(.*)_rounded_fma_k(\d+)',p.stem);r=final(p)
 if m and r:
  ds,k=m.groups();k=int(k);f=four.get((ds,k,'fp16'));a=four.get((ds,k,'sameflow'))
  if f and a:rounded.append(dict(dataset=ds,K=k,rounded_fma_test=r['test'],fp16_test=f['test'],sameflow_test=a['test'],rounded_minus_fp16=r['test']-f['test'],rounded_minus_sameflow=r['test']-a['test']))
csvout('rounding_control.csv',rounded)
chunked=[]
for p in sorted((R/'accumulation_control').glob('*chunked_fp16*.txt')):
 m=re.match(r'(.*)_chunked_fp16_k(\d+)',p.stem);r=final(p)
 if m and r:
  ds,k=m.groups();k=int(k);f=four.get((ds,k,'fp16'));a=four.get((ds,k,'sameflow'))
  if f and a:chunked.append(dict(dataset=ds,K=k,chunked_test=r['test'],fp16_test=f['test'],sameflow_test=a['test'],chunked_train=r['train'],fp16_train=f['train'],sameflow_train=a['train'],chunked_minus_fp16=r['test']-f['test'],chunked_minus_sameflow=r['test']-a['test']))
csvout('accumulation_control.csv',chunked)

extpath=R/'external_time_to_target.csv'
extrows=list(csv.DictReader(extpath.open())) if extpath.exists() else []
lines=['# October review — measurement status','', 'Authoritative inputs: `../whattodo/Review Accelerating ALS Matrix Factorization on GPUs via Mixed-Precision Tensor Cores.pdf` and `../whattodo/GPU_ALS_Matrix_Factorization (2).zip`. Manuscript extracted unchanged into `manuscript/`. The baseline zip was not used.','', 'Primary performance measurements use GPU 1, RTX 3060, sm_86. After the user requested both GPUs, accumulation-reset diagnostics and validation accuracy pairs moved to GPU 0 (also RTX 3060, desktop active); their timings are not performance headline evidence. External runs on GPU 1 overlap those GPU 0 accuracy jobs, sharing the host. Original source and historical results are preserved. Experimental source is in `src/`, BF16 source in `bf16_src/`, external instrumentation in `external_src/`. Timings exclude preprocessing and loading.','', 'Existing PDFs are intermediate and were not refreshed after the user requested no more plotting. Use the current CSVs and generated LaTeX tables; see `figures/STATUS.md`.','', '## Required results and status','',f'- Four-way FP32 atomic / FP32 same-dataflow / TF32 / FP16: {len(rows)}/80 runs collected; `four_way.csv`, raw logs and commands in `four_way/`. Fixed 20 sweeps, FP32 solver, no momentum or giant tier, exact FP32 RMSE every sweep.',f'- Validation-only lambda selection: {len(val)}/4 datasets selected; eight candidates, K=96, fixed 30 sweeps. `validation_selection.csv`. New split is approximately 70/10/20, carving 1/8 of old train into validation with seed 20261014. Original test preserved. Historical exposure to this test cannot be undone by carving validation.',f'- Per-sweep momentum: {len(mom)}/40 curves complete; `momentum_crossings.csv`. Fixed 30-sweep trajectories, beta 0 vs .3, exact RMSE. Both adjacent-sweep and five-sweep-window deltas are reported; these are different stopping rules.',f'- FP64 Gram checks: {len(gram)} per-entity/path records; `gram_errors.csv`. Top 20 users and items per dataset/rank; actual CUDA kernels on the same trained FP32 factor snapshot, upper-triangle errors.',f'- BF16 Gram operands and BF16 solver with refinement: {len(bf)}/8 runs; beta .3 and .5 at K=96 on four datasets. `bf16.csv`. Any zeroed solve makes a configuration unsafe.', f'- Common held-out target curves: {len(extrows)}/84 curves collected in `external/`; all methods use identical warm-validation rows. APR, cuMF-ALS, CCD, and BALS across five ranks; SGD rank128 compared at APR K96. Cold validation rows excluded consistently. SGD uses one continuous model with per-epoch GPU validation and a final saved-model CPU scoring check; epoch barriers differ from the original persistent-kernel schedule.', '- Cumulative ablation: six runs completed in `cumulative/`; one source snapshot. See `CUMULATIVE_PROTOCOL.md` for the switches and combined transitions. Gains are measured, never forced to be monotone. Existing code has no switch for lambda fusion / asynchronous staging, so those two effects are not separately isolated.', '- Nsight long-scoreboard / LG-throttle / giant execution-pipe counters: permission-blocked (`ncu/permission_check.txt`, ERR_NVGPUCTRPERM); administrator script pending; old utilization counters do not establish a bandwidth bottleneck.', '- Solver safety: original FP16 guard zeroes failed solutions; it does NOT retry in FP32. A production FP32 retry remains required before claiming a general safe solver.', '- Optional padded ranks and Yahoo/Amazon: not implemented/not collected. Existing 60,000-entity batches already split datasets whose user counts exceed that size; a larger dataset would add scale coverage.', '', '## Newly gathered numerical findings','']
if rows:
 lines+=['| Dataset | K | atomic / FP16 wall | same-dataflow / FP16 wall | same-dataflow / FP16 compute phase |','|---|---:|---:|---:|---:|']
 for ds,k in sorted({(x['dataset'],x['K']) for x in rows}):
  a=four.get((ds,k,'atomic'));f=four.get((ds,k,'fp16'));s=four.get((ds,k,'sameflow'))
  if a and f and s:lines.append(f"| {ds} | {k} | {a['wall_s']/f['wall_s']:.3f} | {s['wall_s']/f['wall_s']:.3f} | {s['gram_s']/f['gram_s']:.3f} |")
 lines+=['','The compute-phase column sums the legacy User/Item LHS+RHS event intervals. Because these join the overlapping solve streams, they include solver work and MUST NOT be labeled isolated Gram-kernel time. Runs are single measurements, not new variance estimates. The same-dataflow control uses FP32 staging and direct FMA, one warp/entity, 16-row chunks, eight accumulator entries per lane per upper tile, and no Gram atomics. At K<=48, FP16 also uses its existing prefetch path; FP32/TF32 staging is wider and lacks that prefetch. Treat this as a structural dataflow control, not an instruction-identical substitution.','']
if any(r['zeroed_solves'] for r in sweep):
 lines+=['Validation safety failures (retained in `validation_sweep.csv`): '+', '.join(f"{r['run']}: {r['zeroed_solves']} zeroed solves" for r in sweep if r['zeroed_solves'])+'.','']
lines += [f'- Matched BF16/FP16 safety controls: {len(bfc)}/12 additional FP16 runs in `bf16_controls.csv`; FP16 with one refinement at beta .3/.5 and shipped no-refinement FP16 at beta .5. GPU0 accuracy/safety diagnostics, not cross-GPU timing comparisons.']
lines += ['',f'- Selected-recipe test evaluations: {len(held)}/20; `heldout_test.csv`.',f'- New-split controlled FP32/FP16 accuracy pair: {len(accuracy)}/40; `validation_accuracy_pair.csv`. Fixed 30 sweeps, FP32 solver, no momentum/giant tier, validation-selected lambda, test checked only at the end.',f'- Rounded-input FP32 controls: {len(rounded)}/20; `rounding_control.csv`.',f'- Chunk-reset tensor accumulation controls: {len(chunked)}/20; `accumulation_control.csv`. These GPU0 diagnostic timings are not headline performance evidence.','']
if len(mom)==40:
 md={(r['dataset'],r['K'],r['beta']):r for r in mom}
 ratios=[md[ds,k,'0']['first_delta1_wall_s']/md[ds,k,'0.3']['first_delta1_wall_s'] for ds,k,b in md if b=='0']
 lines += [f'Momentum at the adjacent-sweep training-delta threshold improves measured crossing time by {min(ratios):.2f}–{max(ratios):.2f}x across the 20 settings. The five-sweep-window results are a different stopping rule and remain separately reported.','']
if len(bfc)==12:
 lines += ['BF16 result: beta .5 is stable for FP16 on MovieLens, but Netflix FP16 fails with/without refinement (3,142/3,492 zeroed entity-solves; test RMSE 1.440298/1.339762). BF16 with one refinement has zero guards and RMSE 0.817964. This supports the tested BF16 configuration for this failure case, not universal safety or uniformly better accuracy.','']
if len(chunked)==20:
 maxdiff=max(abs(r['chunked_minus_sameflow']) for r in chunked)
 lines += [f'Resetting the tensor accumulator every 16-row chunk and combining chunks with FP32 CUDA-core adds returns test RMSE to within {maxdiff:.6f} of the FP32 control at all 20 dataset/rank points. Rounded operands alone do not reproduce the original FP16 shift. Together with the FP64 Gram errors, this isolates persistent tensor accumulation as the source of the small trajectory/RMSE shift in these experiments; it does not mean FP16 constructs a more accurate Gram matrix. See `NUMERICAL_AUDIT.md` and `accumulation_errors.csv`.','']
if val:
 lines+=['Validation selections: '+', '.join(f"{v['dataset']}: lambda={v['selected_lambda']}, validation RMSE={v['validation_rmse']:.6f}" for v in val)+'.','']
if gram:
 for label in ('atomic_fp32','dense_fp16','sameflow_fp32','giant_fp16'):
  e=[x['relative_upper_frobenius'] for x in gram if x['path']==label]
  lines.append(f'- {label}: relative upper-triangle Frobenius error median {np.median(e):.3g}, maximum {max(e):.3g}.')
if len(extrows)==84:
 lines += ['', '## External target times at the K96 comparison', '', 'Seconds to the preset warm-validation RMSE target. APR/CCD/BALS rank96, cuMF-ALS rank100, SGD rank128. Single runs; shared-host execution and evaluation overhead are documented in `EXTERNAL_PROTOCOL.md`.', '', '| Dataset | Target | APR | ALS | CCD | BALS | SGD |', '|---|---:|---:|---:|---:|---:|---:|']
 for ds in ('ml10m','ml20m','ml32m','netflix'):
  selected={r['method']:r for r in extrows if r['dataset']==ds and r['K'] in ('96','128')}
  values=[f"{float(selected[m]['seconds_to_target']):.4f}" if selected[m]['seconds_to_target'] else 'unreached' for m in ('apr','als','ccd','bals','sgd')]
  lines.append('| '+ds+' | '+selected['apr']['target']+' | '+' | '.join(values)+' |')
 lines += ['', 'APR is 2.75–4.81x faster than cuMF-ALS in this rank pairing at these targets. SGD reaches the target sooner on ML20M and ML32M; Netflix differs by less than 1% and should be treated as a single-run near-tie, not evidence of a reliable ordering. APR reaches the target sooner on ML10M. Final validation accuracy is also reported in `external_time_to_target.csv`; reaching an easier target first does not establish better final accuracy.', '']
lines+=['','## Existing evidence (historical protocol, not new runs)','', '- `../results/PAPER_MEASUREMENTS_20260803/`: 100 variance runs; three-way precision ablation; Nsight utilization/occupancy/registers; full-rank BALS ladder; thresholds; old lambda sweep; coarse cuMF time-to-target. The new manuscript already incorporates much of this.', '- `empty_entity_census.txt`: freshly reproduced cold-item counts ML-10M 30, ML-20M 909, ML-32M 4,182, Netflix 0; affected original test ratings 41, 1,006, 4,723, 0.', '- Old test-selected lambda sweep must not be presented as validation selection. Old TTT has five-sweep resolution. Old precision ablation bundles dataflow and tensor instructions.', '', '## Next / paper integration','', 'Use the completed new controls and validation results to replace the relevant tables; do not mix old 80/20 and new 70/10/20 absolute RMSE. Rebuild the main two-path harness with the validated FP32 control before replacing its headline. Integrate the external comparison first, then the mechanism analysis. The current manuscript is preserved unchanged; requested editorial restructuring, template changes, author confirmation, and repository/Zenodo publication are separate remaining work. No new acceptance or speedup claim should be made from incomplete cells.']
(R/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
print('four_way',len(rows),'momentum',len(mom),'validation',len(val),'gram',len(gram),'bf16',len(bf))
