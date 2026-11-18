# Working with the code

## Recalculate the reported comparisons

Run `python3 results.py` from the root to calculate the headline differences from the original CSV and JSON files in each study. Positive gains favor the intervention. Broad text and arithmetic answer loss are distinct endpoints.

To check the 1.7B findings from individual token scores, run `bash scripts/replay_scores.sh` with Python 3 and NumPy installed. With uv, use `uv run --no-project --python 3.11 --with numpy bash scripts/replay_scores.sh`. The three included programs reconstruct the baseline selection, the two-order comparison with computation supervision and the two-domain mixture decision. Each evidence directory contains its original readout, training histories and checksum manifest. [Baseline](../replay/uniform-lm-baseline-20261022/) · [Computation auxiliary](../replay/scaffold-learning-20261022/) · [Mixture](../replay/uniform-mixture-20261022/)

## Inspect or run a mechanism

| Question | Start here | What to look for |
|---|---|---|
| What information could the frozen selector see? | [Reader gate](../studies/01-useful-thoughts/train/causal_reader_gate.py) · [Fresh evaluation](../studies/01-useful-thoughts/train/reader_confirmation.py) | The gate's inputs and the no-calculation controls used on ordinary text. |
| Was the ordinary source loss held fixed? | [Packing and objective](../studies/01-useful-thoughts/train/scaffold_learning.py) · [Tests](../studies/01-useful-thoughts/tests/test_scaffold_learning.py) | All 528 source tokens precede the work suffix. The tests compare gradients with an independent dense calculation. |
| What does a pause position predict? | [Pause packing and schedule](../studies/02-pause-pilot/train/pause_pretraining.py) · [Tests](../studies/02-pause-pilot/tests/test_pause_pretraining.py) | The last pause predicts the next real token. The late switch must preserve optimizer state. |
| How does a generated state influence the next token? | [CPU sequential forward](../studies/03-autonomous-compute/compute_trainium/interleaved_sequential_reference.py) · [Parity test](../studies/03-autonomous-compute/tests/test_interleaved_sequential_reference.py) | Read a real token, feed its generated vector back through the model, cache both entries, then score the next real token. The test compares cached and recomputed states, logits and gradients and checks that changing a suffix cannot alter earlier scores. |
| How did that model run on Trainium? | [Grouped model implementation](../studies/03-autonomous-compute/compute_trainium/grouped_interleaved_workspace.py) · [Equal-data driver](../studies/03-autonomous-compute/scripts/run_latent_exposure_screen.py) · [Matched-work driver](../studies/03-autonomous-compute/scripts/recover_latent_exposure_work.py) | Trace the schedule, optimizer checks, real targets and work ledger from experiment to report. |
| How were the last four conditions assigned? | [Factorial driver](../studies/03-autonomous-compute/scripts/run_calculation_factorial.py) · [Registered plan](../studies/03-autonomous-compute/results/calculation_exposure_20261030/factorial-v1/plan.json) | Original and enriched source distributions were each assigned to both model families. |

## Run the CPU checks

From the top of a regular clone, run `bash scripts/check_cpu.sh`. The script uses uv, Python 3.11, PyTorch 2.9.1 and Transformers 5.15.1. It runs the cached sequential-state tests, pause packing and recovery tests, and the supervised objective tests directly against the bundled study implementations. Two pause-cache checks fail with the older Transformers 4.57.6 cache API. [Test runner](../scripts/check_cpu.sh) · [Recorded pause environment](../studies/02-pause-pilot/validation/qualification_20261022/environment.json)

## Native runs and model weights

Native training requires the matching Neuron runtime, Trainium hardware, training data and checkpoint storage. Original launch scripts encode the research node's paths, topology and recovery state. Their invocation and environment assumptions are documented in the study directories. Exact retraining of the final from-scratch comparison also requires the private source windows and native state. [Trainer guide](../studies/03-autonomous-compute/docs/TRAINER.md) · [Data and storage](../studies/03-autonomous-compute/docs/DATA.md)

The 1.7B study released twelve lossless model deltas. Use mixed-seed11 or mixed-seed12 for the two-domain result and warmup_cosine-lr1e-05 for the selected all-math model. The restore code and license are included here. Running inference requires the external base checkpoint and the corresponding weight delta. The catalog gives their exact revisions, checksums and reconstruction command. [Restore utility](../tools/restore-model/restore_model_delta.py) · [Model catalog](../studies/01-useful-thoughts/docs/MODELS_20261022.md)
