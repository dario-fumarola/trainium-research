# Sources and scope

This handoff covers the October Trainium research commits from October 16 through the last recorded update on October 30, 2026. The three public source repositories are pinned as Git submodules. A recursive clone retrieves their original code, tests, protocols and result archives. The parent repository holds a small CPU example and selected aggregate evidence directly.

| Source | Pinned revision | Role |
|---|---|---|
| useful-thoughts-trainium | [d07d89e034](https://github.com/dario-fumarola/useful-thoughts-trainium/tree/d07d89e0348cf1acd76acd1173a90dcd11facf3e) | Pretrained 1.7B controlled experiments and releases |
| pause-pretraining-trainium | [a7d6a14c30](https://github.com/dario-fumarola/pause-pretraining-trainium/tree/a7d6a14c30b12c5c6a920f04c1c1b5121065afa2) | Superseded 0.6B hardware qualification |
| autonomous-compute-trainium | [15d52172a8](https://github.com/dario-fumarola/autonomous-compute-trainium/tree/15d52172a8773c905d9951641ea9c0cbc3451fd5) | Natural-text pretraining, recurrence and workspace studies |

The older August–September EMA snapshot and local September review bundles are predecessors. Those pilots examined whether a reader could use a sampled note and whether ordinary language prediction suffered. Their runs, private archives and machine records are outside the October handoff.

## Files included in a standard clone

Each entry below was checked byte for byte against its pinned source. The CSV files retain full precision. The JSON files preserve the original scores and decision fields.

| Local file | Original |
|---|---|
| [scaffold.csv](../evidence/scaffold.csv) | [Source](https://github.com/dario-fumarola/useful-thoughts-trainium/blob/d07d89e0348cf1acd76acd1173a90dcd11facf3e/validation/scaffold_learning_complete_20261022/absolute-scores.csv) |
| [source-mixture.csv](../evidence/source-mixture.csv) | [Source](https://github.com/dario-fumarola/useful-thoughts-trainium/blob/d07d89e0348cf1acd76acd1173a90dcd11facf3e/validation/uniform_mixture_complete_20261022/absolute-scores.csv) |
| [pause-closure.json](../evidence/pause-closure.json) | [Source](https://github.com/dario-fumarola/pause-pretraining-trainium/blob/a7d6a14c30b12c5c6a920f04c1c1b5121065afa2/validation/closure_20261022/summary.json) |
| [equal-data-arithmetic.json](../evidence/equal-data-arithmetic.json) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/results/latent_exposure_20261030/learning-v1/equal-data-arithmetic.json) |
| [equal-data-broad.json](../evidence/equal-data-broad.json) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/results/latent_exposure_20261030/learning-v1/equal-data-broad.json) |
| [matched-work-recovery-v1-arithmetic.json](../evidence/matched-work-recovery-v1-arithmetic.json) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/results/latent_exposure_20261030/work-recovery-v1/matched-work-recovery-v1-arithmetic.json) |
| [matched-work-recovery-v1-broad.json](../evidence/matched-work-recovery-v1-broad.json) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/results/latent_exposure_20261030/work-recovery-v1/matched-work-recovery-v1-broad.json) |
| [__init__.py](../examples/latent/compute_trainium/__init__.py) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/compute_trainium/__init__.py) |
| [interleaved_sequential_reference.py](../examples/latent/compute_trainium/interleaved_sequential_reference.py) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/compute_trainium/interleaved_sequential_reference.py) |
| [interleaved_workspace.py](../examples/latent/compute_trainium/interleaved_workspace.py) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/compute_trainium/interleaved_workspace.py) |
| [model.py](../examples/latent/compute_trainium/model.py) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/compute_trainium/model.py) |
| [probability_workspace.py](../examples/latent/compute_trainium/probability_workspace.py) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/compute_trainium/probability_workspace.py) |
| [untied_probability_workspace.py](../examples/latent/compute_trainium/untied_probability_workspace.py) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/compute_trainium/untied_probability_workspace.py) |
| [test_interleaved_sequential_reference.py](../examples/latent/tests/test_interleaved_sequential_reference.py) | [Source](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/tests/test_interleaved_sequential_reference.py) |

## Historical branches

All published branch tips of the pause and autonomous-compute repositories are ancestors of their pinned main revisions. Six useful-thoughts development tips have commits outside main. Their exact snapshots are listed here for anyone investigating the preparatory work. The experiment guide uses the terminal reports for outcomes.

- [candidate-utility-v30](https://github.com/dario-fumarola/useful-thoughts-trainium/tree/a84c77b89fd469e40536353b8744856136984e03)
- [causal-decoder-batching-v1](https://github.com/dario-fumarola/useful-thoughts-trainium/tree/b25560a9618970a43afab349edc785d772a02604)
- [document-evidence-v12](https://github.com/dario-fumarola/useful-thoughts-trainium/tree/7c5c26414945e3f1c7c937bfac98a13762d4c7b0)
- [general-sft-v5](https://github.com/dario-fumarola/useful-thoughts-trainium/tree/9e3f4b298d4a4870e3be4844bc9c34de2f4dfef5)
- [real-prefix-partial-v9](https://github.com/dario-fumarola/useful-thoughts-trainium/tree/9d57df721e81fa88b3f87ab2766184b9b38ae819)
- [variable-depth-eval-v11](https://github.com/dario-fumarola/useful-thoughts-trainium/tree/11512db569ce62dd844c9bb530149018413826c3)

Source code and materials retain their original notices. Models and datasets have their own terms. [Research repository notices](https://github.com/dario-fumarola/useful-thoughts-trainium/blob/d07d89e0348cf1acd76acd1173a90dcd11facf3e/NOTICE.md) · [Third-party material](https://github.com/dario-fumarola/autonomous-compute-trainium/tree/15d52172a8773c905d9951641ea9c0cbc3451fd5/third_party)
