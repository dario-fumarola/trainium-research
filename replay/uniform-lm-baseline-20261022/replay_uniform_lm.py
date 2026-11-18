"""Replay source-free native scores with NumPy; no model or source text needed.

Usage: python replay_uniform_lm.py PUBLIC_EVIDENCE_DIRECTORY
This is an independent publication check, not the production statistics path.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np


def require(test, message):
    if not test:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.allclose(actual, expected, rtol=0, atol=5e-13), message)


root = Path(sys.argv[1])
readout = json.loads((root/'readout.json').read_text())
array_file = root/'original-token-scores.npz'
with array_file.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
require(digest == readout['public_scores_sha256'], 'Score archive SHA256 differs')
choice = json.loads((root/'frozen-selection.json').read_text())
terminals = json.loads((root/'all-terminal-models.json').read_text())['models']
require(len(terminals) == 4 and all(v['step'] == 128 for v in terminals.values()),
        'Exactly four terminal128 models required')
require(choice['recipe'] == readout['selected_recipe'], 'Selected recipe differs')
require(choice['checkpoint'] == terminals[choice['recipe']], 'Selected checkpoint differs')
absolute = {}
contrasts = 0
cohort_documents = []
with np.load(array_file, allow_pickle=False) as archive:
    require(set(archive.files) == set(readout['arrays']) | {
        c+'__'+s for c in ('DEV', 'CONFIRM', 'CALCULATION')
        for s in ('item_ids', 'document_ids')}, 'Unspecified or missing score arrays')
    for cohort, expected_n in (('DEV', 1024), ('CONFIRM', 4096), ('CALCULATION', 1024)):
        ids, docs = (archive[cohort+'__'+key].tolist() for key in ('item_ids', 'document_ids'))
        require(len(ids) == len(set(ids)) == len(docs) == len(set(docs)) == expected_n,
                'Complete unique source pairing required')
        cohort_documents.append(set(docs))
        models = {v['model']: archive[k].astype(np.float64) for k, v in
                  readout['arrays'].items() if v['cohort'] == cohort}
        expected_models = {'B0'} | (set(terminals) if cohort == 'DEV' else {choice['recipe']})
        require(set(models) == expected_models, 'Frozen model inventory differs')
        absolute[cohort] = {}
        for name, score in models.items():
            require(score.shape == (expected_n, 527) and np.isfinite(score).all(),
                    'Complete finite original-token scores required')
            current = dict(full_window_NLL=float(-score.mean()),
                suffix16_NLL=float(-score[:, -16:].mean()),
                suffix16_log_probability=float(score[:, -16:].sum(axis=1).mean()))
            absolute[cohort][name] = current
            for key, value in current.items():
                close(value, readout['metrics'][cohort]['absolute'][name][key],
                      'Absolute score mismatch: '+cohort+'/'+name+'/'+key)
            if name == 'B0':
                continue
            d = score-models['B0']
            per_document = np.column_stack((d.mean(axis=1), d[:, -16:].sum(axis=1)))
            # Each source is unique: resampling documents carries every token
            # together. Match the declared Generator stream in blocks of 32.
            rng = np.random.default_rng(17)
            resamples = []
            for first in range(0, 5000, 32):
                indices = rng.integers(0, expected_n,
                    size=(min(32, 5000-first), expected_n))
                resamples.extend(per_document[indices].sum(axis=1)/expected_n)
            mean = per_document.sum(axis=0)/expected_n
            interval = np.quantile(np.asarray(resamples), [.025, .975], axis=0).T
            reported = readout['metrics'][cohort]['contrasts'][name]
            close(mean, [reported['full_window_gain_nats_per_token'],
                         reported['suffix16_gain_nats']], 'Paired mean mismatch')
            close(interval, [reported['full_window_CI95'], reported['suffix16_CI95']],
                  'Paired document confidence interval mismatch')
            require(reported['statistics']['independent_clusters'] == expected_n
                    and reported['statistics']['bootstrap_draws'] == 5000
                    and reported['statistics']['bootstrap_seed'] == 17
                    and reported['statistics']['resampling_unit'] == 'unique source document',
                    'Declared resampling unit or stream differs')
            contrasts += 1
require(all(not a & b for i, a in enumerate(cohort_documents)
            for b in cohort_documents[i+1:]), 'Evaluation source cohorts overlap')
chosen = min(terminals, key=lambda n: (absolute['DEV'][n]['full_window_NLL'],
                                     terminals[n]['lr'], n))
require(chosen == choice['recipe'], 'Frozen DEV selection rule differs')
passes = readout['metrics']['CONFIRM']['contrasts'][chosen]['full_window_CI95'][0] > 0
require(passes == readout['uniform_baseline_improves']
        and readout['verdict'] == ('PASS' if passes else 'FAIL'), 'Scientific verdict differs')
print(json.dumps(dict(status='PASS_SOURCE_FREE_NUMPY_REPLAY',
    documents_per_cohort=[1024, 4096, 1024], full_token_scores_preserved=True,
    contrasts_reproduced=contrasts, selected_recipe=chosen,
    confirmation_verdict=readout['verdict'],
    scope='Numerical reconstruction from released scores; no backbone rerun, original source verification or external replication')))
