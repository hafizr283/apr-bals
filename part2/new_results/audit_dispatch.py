import struct,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
rows=[]
for ds,stem in {'ml10m':'ratings10','ml20m':'ratings','ml32m':'ratings32','netflix':'netflix_ratings'}.items():
 p=R/'splits'/f'{stem}_validation.bin'
 with open(p,'rb') as f:
  f.seek(20);vectors=[]
  for i in range(12):
   n=struct.unpack('<Q',f.read(8))[0]
   if i in (6,9):vectors.append(np.fromfile(f,'<i4',n))
   else:f.seek(n*4,1)
 for side,offset in zip(('users','items'),vectors):
  d=np.diff(offset);nd=int((d>=4).sum())//32*32;ng=min(int((d>=1024).sum()),nd)
  rows.append(dict(dataset=ds,side=side,dense_prefix_size=nd,dense_prefix_below_threshold=int((d[:nd]<4).sum()),sparse_suffix_above_threshold=int((d[nd:]>=4).sum()),giant_prefix_size=ng,giant_prefix_below_threshold=int((d[:ng]<1024).sum()),nongiant_suffix_above_threshold=int((d[ng:]>=1024).sum())))
(R/'dispatch_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print(rows)
