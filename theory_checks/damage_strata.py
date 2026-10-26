import pandas as pd, numpy as np
def fp(p):
    if p>=0.01: return f'{p:.3f}' if p<0.1 else f'{p:.2f}'
    m,e=f'{p:.1e}'.split('e'); return f'${m}\\cdot10^{{{int(e)}}}$'
from scipy.stats import wilcoxon
R='results/v2/'
dg=pd.read_csv(R+'dataset_kfold_ext/diagnostics.csv')
m=pd.read_csv(R+'dataset_kfold/results.csv',low_memory=False)
key=['dataset','attack','rate','seed']
piv=m[m.seed==0].pivot_table(index=key,columns='method',values='AUPRC')
x=dg[key+['damage_rf','damage_proxy']].merge(piv.reset_index(),on=key)
BASE=['IsoForest','LOF','OCSVM','SpecSig','GradNorm','SEVER','GradSpectral','kNNLabel','kNNDist','ConfidentLearning','ECOD','IsoForest+CL','AE-Recon','ECOD+CL']
SH={'ECOD':'ECOD','ConfidentLearning':'CL','kNNLabel':'$k$-NN label','GradNorm':'GradNorm','IsoForest':'IForest'}
lab=['$\\le0.5$','0.5--2','2--5','$>5$']
out=[]
for dmg,name in [('damage_rf','Random forest'),('damage_proxy','Logistic proxy')]:
    x['bin']=pd.cut(x[dmg],[-1,0.005,0.02,0.05,1],labels=lab)
    first=True
    for b,g in x.groupby('bin',observed=True):
        means=g[BASE].mean(); best=means.idxmax()
        pdm=g.groupby('dataset')[list(dict.fromkeys(['RPFNet',best]))].mean()
        p=wilcoxon(pdm.RPFNet,pdm[best]).pvalue
        out.append(f"{name if first else ''} & {b} & {len(g)} & {g.dataset.nunique()} & {g.RPFNet.mean():.3f} & {SH.get(best,best)} {means.max():.3f} & {g.ECOD.mean():.3f} & {g.ConfidentLearning.mean():.3f} & {int((pdm.RPFNet>pdm[best]).sum())}/{len(pdm)} & {fp(p)}\\\\")
        first=False
    out.append('\\midrule')
out=out[:-1]
open('tab_damage.tex','w').write('\n'.join(out)+'\n')
print('\n'.join(out))
print('share rf<=0.5pt: %.2f'%(x.damage_rf<=0.005).mean(), 'share rf>2pt %.3f'%(x.damage_rf>0.02).mean())
