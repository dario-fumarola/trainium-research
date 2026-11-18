"""NumPy-only replay of all original direct-prediction scores and decisions."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

root=Path(sys.argv[1]);r=json.loads((root/'readout.json').read_text());path=root/'original-token-scores.npz'
def require(condition,message):
 if not condition:raise ValueError(message)
def close(a,b,message):require(np.allclose(a,b,rtol=0,atol=5e-13),message)
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
require(digest(path)==r['public_scores_sha256'],'Original-score archive SHA differs')
models=r['models'];expected={f'scaffold_{a}-seed{s}' for a in ('raw','format','trace') for s in (11,12)}
require(set(models)==expected|{'uniform_reference'} and all(v['step']==128 for v in models.values()),'Every prespecified terminal model must remain')
require(r['all_six_frozen_before_confirmation'],'All six must freeze before confirmation')
for name,sha in r['original_training_history_sha256'].items():
 f=root/'training-history'/(name+'.jsonl');require(digest(f)==sha,'Original native history SHA differs')
 history=[json.loads(x) for x in f.open()]
 require([v['step'] for v in history]==list(range(1,129)),'Every original optimizer step must remain')
 seen=[item for v in history for item in v['ordered_item_ids']]
 require(len(seen)==len(set(seen))==34816,'Every fitting source must occur once')
 require(all(v['loss_components']['source_labels']==143344 and v['gradient_reductions']==1 for v in history),'Actual source target count or reduction differs')
seen_docs=[];contrast_count=0
with np.load(path,allow_pickle=False) as archive:
 require(set(archive.files)==set(r['arrays'])|{c+'__'+s for c in ('CONFIRM','CALCULATION') for s in ('item_ids','document_ids')},'Hidden or missing arrays')
 for cohort,n in [('CONFIRM',4096),('CALCULATION',1024)]:
  ids=archive[cohort+'__item_ids'].tolist();docs=archive[cohort+'__document_ids'].tolist()
  require(len(ids)==len(set(ids))==len(docs)==len(set(docs))==n,'Unique complete source pairing required');seen_docs.append(set(docs))
  scores={meta['model']:archive[key].astype(np.float64) for key,meta in r['arrays'].items() if meta['cohort']==cohort}
  require(set(scores)==set(models)|{'B0'},'Every model must remain in the readout')
  for name,v in scores.items():
   require(v.shape==(n,527) and np.isfinite(v).all() and (v<=1e-6).all(),'Complete finite original-token probabilities required')
   close(-v.mean(),r['metrics'][cohort]['absolute'][name]['full_window_NLL'],'Absolute full-window NLL differs')
   close(v[:,-16:].sum(1).mean(),r['metrics'][cohort]['absolute'][name]['suffix16_log_probability'],'Suffix score differs')
  contrasts=r['metrics'][cohort]['contrasts'];columns=[]
  expected_pairs={(name,'B0') for name in models}
  expected_pairs|={(f'scaffold_trace-seed{s}',f'scaffold_{a}-seed{s}') for s in (11,12) for a in ('raw','format')}
  expected_pairs|={(f'scaffold_trace-seed{s}','uniform_reference') for s in (11,12)}
  require({(v['treatment'],v['control']) for v in contrasts.values()}==expected_pairs,'A prespecified contrast is missing')
  for key,c in contrasts.items():
   delta=scores[c['treatment']]-scores[c['control']];columns.extend([delta.mean(1),delta[:,-16:].sum(1)])
   require(c['statistics']['independent_clusters']==n and c['statistics']['bootstrap_draws']==5000 and c['statistics']['bootstrap_seed']==17 and c['statistics']['resampling_unit']=='unique source document','Independent resampling unit or stream differs')
  values=np.column_stack(columns);rng=np.random.default_rng(17);samples=[]
  for first in range(0,5000,32):
   indexes=rng.integers(0,n,size=(min(32,5000-first),n));samples.extend(values[indexes].sum(1)/n)
  means=values.sum(0)/n;interval=np.quantile(np.asarray(samples),[.025,.975],axis=0).T
  for index,(key,c) in enumerate(contrasts.items()):
   close(means[index*2:index*2+2],[c['full_window_gain_nats_per_token'],c['suffix16_gain_nats']],'Paired means differ')
   close(interval[index*2:index*2+2],[c['full_window_CI95'],c['suffix16_CI95']],'Paired document intervals differ');contrast_count+=1
require(not seen_docs[0]&seen_docs[1],'Fresh natural and old calculation cohorts overlap')
cs=r['metrics']['CONFIRM']['contrasts']
def positive(t,c):return cs[t+'__minus__'+c]['full_window_CI95'][0]>0
content=all(positive(f'scaffold_trace-seed{s}',f'scaffold_{a}-seed{s}') for s in (11,12) for a in ('raw','format'))
base=all(positive(f'scaffold_trace-seed{s}','B0') for s in (11,12));qualified=all(positive(f'scaffold_trace-seed{s}','uniform_reference') for s in (11,12))
require(content==r['content_supervision_improves_both_orders'] and base==r['trace_improves_B0_both_orders'] and qualified==r['trace_beats_qualified_uniform_reference_both_orders'] and (content and base)==r['primary_pass'],'Scientific decision differs')
print(json.dumps(dict(status='PASS_SOURCE_FREE_NUMPY_REPLAY',contrasts_reproduced=contrast_count,primary_pass=content and base,content_both_orders=content,beats_B0_both_orders=base,beats_qualified_uniform_both_orders=qualified,source_scores_exact=True,scope='Within-repository numerical reconstruction, not backbone execution, source verification or external replication.')))
