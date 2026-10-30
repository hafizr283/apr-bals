#!/usr/bin/env python3
import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parent
files={}
for folder in ('src','bf16_src','external_src','chunked_src','bin'):
 for p in sorted((R/folder).rglob('*')):
  if p.is_file():files[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'artifact_sha256.json').write_text(json.dumps(files,indent=2)+'\n')
checks=[]
for group in ('four_way','momentum','validation','heldout_test','rounding_control','bf16','bf16_controls','cumulative','accumulation_control','validation_accuracy_pair'):
 for p in sorted((R/group).glob('*.txt')):
  s=p.read_text();meta=p.with_suffix('.json')
  checks.append({'log':str(p.relative_to(R)),'final_present':'=== APR-BALS final ===' in s,'metadata_present':meta.exists(),'nonfinite_text':bool(re.search(r'\b(?:nan|inf)\b',s,re.I)),'zeroed_solves':list(map(int,re.findall(r'non-finite solves zeroed = (\d+)',s)))})
(R/'artifact_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
print(len(files),'hashed artifacts;',len(checks),'run logs audited')
for r in checks:
 if not r['final_present'] or not r['metadata_present'] or r['nonfinite_text'] or any(r['zeroed_solves']):print(r)
