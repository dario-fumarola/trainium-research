"""Standalone NumPy reconstruction of both domains and fixed50/50 target."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
D=Path(sys.argv[1]);r=json.loads((D/'readout.json').read_text())
def require(test,message):
 if not test:raise ValueError(message)
def close(a,b,message):require(np.allclose(a,b,rtol=0,atol=5e-13),message)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
require(sha(D/'original-token-scores.npz')==r['public_scores_sha256'],'Original scores changed')
models={'mixed-seed11','mixed-seed12','uniform_reference'};require(set(r['all_terminal_models'])==models and all(v['step']==128 for v in r['all_terminal_models'].values()),'Every fixed terminal must remain');require(r['both_models_fixed_before_fresh_confirmation'],'Both terminals must precede confirmation')
require(r['plan']['target_mixture']['math']==.5 and r['plan']['target_mixture']['English']==.5,'The declared target mixture changed')
training=[]
for name in ['mixed-seed11','mixed-seed12']:
 file=D/'training-history'/(name+'.jsonl');require(sha(file)==r['original_training_history_sha256'][name],'Original training history changed');history=[json.loads(x) for x in file.open()];require([v['step'] for v in history]==list(range(1,129)),'Every update must remain');ids=[i for v in history for i in v['ordered_item_ids']];require(len(ids)==len(set(ids))==32768 and history[-1]['counters']=={'expressions':32768,'supervised_tokens':17268736},'Exactly one complete source pass required');require(all(v['gradient_reductions']==1 for v in history),'Exactly one reducer required');training.append(set(ids))
require(training[0]==training[1],'Both orders must use identical source exposure')
training_docs={i.split(':',1)[0] for i in training[0]};domains={};draws={};cohort_docs=[];count=0
with np.load(D/'original-token-scores.npz',allow_pickle=False) as a:
 require(set(a.files)==set(r['arrays'])|{c+'__'+k for c in ['MATH_CONFIRM','ENGLISH_CONFIRM'] for k in ['item_ids','document_ids']},'Missing or extra score arrays')
 for cohort,n,seed in [('MATH_CONFIRM',2048,17),('ENGLISH_CONFIRM',1024,19)]:
  itemids=a[cohort+'__item_ids'].tolist();docs=a[cohort+'__document_ids'].tolist();require(len(itemids)==len(set(itemids))==len(docs)==len(set(docs))==n,'Complete unique source pairing required');require(not set(docs)&training_docs,'Confirmation overlaps fitting sources');cohort_docs.append(set(docs))
  scores={v['model']:a[k].astype(np.float64) for k,v in r['arrays'].items() if v['cohort']==cohort};require(set(scores)==models|{'B0'},'All absolute controls must remain')
  for name,v in scores.items():
   require(v.shape==(n,527) and np.isfinite(v).all() and (v<=1e-6).all(),'Finite complete original token scores required');close(-v.mean(),r['metrics'][cohort]['absolute'][name]['full_window_NLL'],'Absolute NLL differs');close(v[:,-16:].sum(1).mean(),r['metrics'][cohort]['absolute'][name]['suffix16_log_probability'],'Final16 score differs')
  contrasts=r['metrics'][cohort]['contrasts'];expected={(m,'B0') for m in models}|{(m,'uniform_reference') for m in ['mixed-seed11','mixed-seed12']};require({(v['treatment'],v['control']) for v in contrasts.values()}==expected,'Every prespecified comparison required');columns=[]
  for v in contrasts.values():
   delta=scores[v['treatment']]-scores[v['control']];columns.extend([delta.mean(1),delta[:,-16:].sum(1)]);s=v['statistics'];require(s['bootstrap_draws']==5000 and s['bootstrap_seed']==seed and s['independent_clusters']==n and s['resampling_unit']=='unique source document','Declared document bootstrap differs')
  values=np.column_stack(columns);mean=values.sum(0)/n;rng=np.random.default_rng(seed);replicates=[]
  for first in range(0,5000,32):
   idx=rng.integers(0,n,size=(min(32,5000-first),n));replicates.extend(values[idx].sum(1)/n)
  replicates=np.asarray(replicates);ci=np.quantile(replicates,[.025,.975],axis=0).T;draws[cohort]=replicates;domains[cohort]=mean
  for i,(key,v) in enumerate(contrasts.items()):close(mean[2*i:2*i+2],[v['full_window_gain_nats_per_token'],v['suffix16_gain_nats']],'Paired means differ');close(ci[2*i:2*i+2],[v['full_window_CI95'],v['suffix16_CI95']],'Paired intervals differ');count+=1
require(not cohort_docs[0]&cohort_docs[1],'The two source strata overlap')
mixture=r['equal_domain_mixture'];keys=list(r['metrics']['MATH_CONFIRM']['contrasts']);require(list(mixture)==keys and list(r['metrics']['ENGLISH_CONFIRM']['contrasts'])==keys,'Stratum comparisons differ')
combined=.5*draws['MATH_CONFIRM']+.5*draws['ENGLISH_CONFIRM'];mean=.5*domains['MATH_CONFIRM']+.5*domains['ENGLISH_CONFIRM'];ci=np.quantile(combined,[.025,.975],axis=0).T
for i,(key,v) in enumerate(mixture.items()):
 close(mean[2*i:2*i+2],[v['full_window_gain_nats_per_token'],v['suffix16_gain_nats']],'Declared-mixture mean differs');close(ci[2*i:2*i+2],[v['full_window_CI95'],v['suffix16_CI95']],'Stratified mixture interval differs');s=v['statistics'];require(s['domain_weights']==[.5,.5] and s['math_seed']==17 and s['English_seed']==19 and s['bootstrap_draws']==5000,'Mixture weights or independent streams differ');count+=1
both=all(r['metrics'][c]['contrasts'][m+'__minus__B0']['full_window_CI95'][0]>0 for c in ['MATH_CONFIRM','ENGLISH_CONFIRM'] for m in ['mixed-seed11','mixed-seed12']);strong=all(mixture[m+'__minus__uniform_reference']['full_window_CI95'][0]>0 for m in ['mixed-seed11','mixed-seed12']);require(both==r['both_orders_improve_both_domains_vs_B0'] and strong==r['both_orders_improve_declared_mixture_vs_qualified_reference'] and (both and strong)==r['primary_pass'],'The scientific decision differs')
print(json.dumps({'status':'PASS_SOURCE_FREE_NUMPY_REPLAY','contrasts_reproduced':count,'both_domains_both_orders':both,'beats_qualified_reference_on_declared_mixture_both_orders':strong,'primary_pass':both and strong,'scope':'Original-score numerical reconstruction, not a backbone rerun, source-text verification or external replication.'}))
