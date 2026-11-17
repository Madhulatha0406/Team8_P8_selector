import json, pandas as pd, numpy as np, sys
sys.path.insert(0,'../scripts')
from common import TABLES, OPENMP, GPU_OWN, ALL
f=pd.read_csv(TABLES/'matrix_features.csv').set_index('matrix_id')
pm=pd.read_csv(TABLES/'performance_matrix.csv').set_index('matrix_id')
cfg=pd.read_csv(TABLES/'unified_config_summary.csv')
mc=pd.read_csv(TABLES/'integrated_mpi_comm.csv'); mc=mc[mc.ranks==8]
cmp_=pd.read_csv(TABLES/'integrated_comm_vs_winner_8ranks.csv').set_index('matrix_id')
m={}
for d in f.index:
    r=f.loc[d]
    omp=cfg[(cfg.dataset==d)&cfg.candidate.isin(OPENMP)].kernel_ms_median.min()
    g=cfg[(cfg.dataset==d)&cfg.candidate.isin(GPU_OWN)][['candidate','kernel_ms_median','transfer_ms_median']].values.tolist()
    par=[p for p in g if not any(q[1]<=p[1] and q[2]<=p[2] and (q[1]<p[1] or q[2]<p[2]) for q in g)]
    par=sorted({(p[0],round(p[1],5),round(p[2],4)) for p in par})
    c=mc[mc.dataset==d].set_index('candidate').total_bytes_sent/1e6
    m[d]=dict(rows=int(r.rows),nnz=int(r.nnz),mean=round(r.mean_nnz_row,2),std=round(r.std_nnz_row,2),max=int(r.max_nnz_row),
      bw=int(r.bandwidth),perf={k:(None if pd.isna(v) else round(v,3)) for k,v in pm.loc[d].items()},omp_ms=round(omp,5),
      gpu=par,comm={k:round(v,3) for k,v in c.items()},fast8=cmp_.loc[d,'fastest_scheme'],minc8=cmp_.loc[d,'min_comm_scheme'])
data=json.dumps(dict(m=m,cands=ALL),separators=(',',':'))
html=open('template.html').read().replace('__DATA__',data)
open('/mnt/user-data/outputs/spmv_suite.html','w').write(html); open('spmv_suite.html','w').write(html)
print(len(html)//1024,'KB')
