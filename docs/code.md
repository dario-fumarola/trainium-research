# Working with the code

## Recalculate the reported comparisons

From the repository root, run `python3 results.py`. It reads the original two score tables, four JSON readouts and pause closure record under `evidence/`. It recomputes differences between absolute losses and sums the assigned hours across all five pause attempts. Positive differences favor the intervention. Broad text and arithmetic answer loss are distinct endpoints. The script operates on saved aggregates. Full per-document uncertainty analyses and training histories live with the original studies. [File provenance](sources.md)

## Inspect or run a mechanism

| Question | Start here | What to look for |
|---|---|---|
| What information could the frozen selector see? | [Reader gate](https://github.com/dario-fumarola/useful-thoughts-trainium/blob/d07d89e0348cf1acd76acd1173a90dcd11facf3e/train/causal_reader_gate.py) · [Fresh evaluation](https://github.com/dario-fumarola/useful-thoughts-trainium/blob/d07d89e0348cf1acd76acd1173a90dcd11facf3e/train/reader_confirmation.py) | The gate's inputs and the no-calculation controls used on ordinary text. |
| Was the ordinary source loss held fixed? | [Packing and objective](https://github.com/dario-fumarola/useful-thoughts-trainium/blob/d07d89e0348cf1acd76acd1173a90dcd11facf3e/train/scaffold_learning.py) · [Tests](https://github.com/dario-fumarola/useful-thoughts-trainium/blob/d07d89e0348cf1acd76acd1173a90dcd11facf3e/tests/test_scaffold_learning.py) | All 528 source tokens precede the work suffix. The tests compare gradients with an independent dense calculation. |
| What does a pause position predict? | [Pause packing and schedule](https://github.com/dario-fumarola/pause-pretraining-trainium/blob/a7d6a14c30b12c5c6a920f04c1c1b5121065afa2/train/pause_pretraining.py) · [Tests](https://github.com/dario-fumarola/pause-pretraining-trainium/blob/a7d6a14c30b12c5c6a920f04c1c1b5121065afa2/tests/test_pause_pretraining.py) | The last pause predicts the next real token. The late switch must preserve optimizer state. |
| How does a generated state influence the next token? | [CPU sequential forward](../examples/latent/compute_trainium/interleaved_sequential_reference.py) · [Parity test](../examples/latent/tests/test_interleaved_sequential_reference.py) | Read a real token, feed its generated vector back through the model, cache both entries, then score the next real token. The test compares cached and recomputed states, logits and gradients and checks that changing a suffix cannot alter earlier scores. |
| How did that model run on Trainium? | [Grouped model implementation](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/compute_trainium/grouped_interleaved_workspace.py) · [Equal-data driver](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/scripts/run_latent_exposure_screen.py) · [Matched-work driver](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/scripts/recover_latent_exposure_work.py) | Trace the schedule, optimizer checks, real targets and work ledger from experiment to report. |
| How were the last four conditions assigned? | [Factorial driver](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/scripts/run_calculation_factorial.py) · [Registered plan](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/results/calculation_exposure_20261030/factorial-v1/plan.json) | Original and enriched source distributions were each assigned to both model families. |

The standalone example preserves the upstream code and test byte for byte. The tiny tensors in the test exercise the computation on a laptop. To run the example with Python and uv:

```sh
cd examples/latent
OMP_NUM_THREADS=1 uv run --no-project --python 3.11 --with torch==2.9.1 --with numpy --with pytest python -m pytest -q
```

For focused CPU checks across all three projects, run these commands from the repository root:

```sh
git submodule update --init
bash scripts/check_cpu.sh
```

The suite covers the cached latent model, the archived pause implementation and the supervised computation objective. The pause cache relies on the Transformers 5.15.1 API. Two cache tests failed under 4.57.6 and passed with the archived version. Native training additionally requires the matching Neuron runtime, Trainium hardware and correctly provisioned data and checkpoints. The historical launch scripts assume that environment. [Pause environment receipt](https://github.com/dario-fumarola/pause-pretraining-trainium/blob/a7d6a14c30b12c5c6a920f04c1c1b5121065afa2/validation/qualification_20261022/environment.json) · [Native trainer guide](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/docs/TRAINER.md)

## Use a released model or replay original scores

The 1.7B study released twelve lossless model deltas. For the two-domain comparison, start with mixed-seed11 or mixed-seed12. For the strongest all-math run, use warmup_cosine-lr1e-05. Download the exact base revision, one delta and the restore tools listed in the catalog. Follow its checksum, reconstruction and standard Hugging Face loading instructions. The numerical archives in the release index provide a smaller route to checking the statistics without restoring model weights. [Model catalog and commands](https://github.com/dario-fumarola/useful-thoughts-trainium/blob/d07d89e0348cf1acd76acd1173a90dcd11facf3e/docs/MODELS_20261022.md) · [Numerical release index](https://github.com/dario-fumarola/useful-thoughts-trainium/releases/tag/overnight-results-20261022)

The final from-scratch study publishes implementation, protocols, numerical audits and aggregate scores. Its raw source text, native model weights, optimizer state and accelerator caches were retained outside Git. Reproducing training from those exact private inputs requires access to the retained research environment. [Data rules](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/docs/DATA.md) · [Matched-work audit](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/results/latent_exposure_20261030/work-recovery-v1/REPORT.md)
