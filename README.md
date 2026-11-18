# What extra computation bought us on Trainium

October 16–30, 2026 · Research handoff

We tested whether a language model can use extra steps before predicting the next token. Those steps took three forms: writing a calculation the model could read, processing blank positions, or generating an internal state that later positions could reuse.

**Writing out a calculation helped on targeted problems. Ordinary next-token training was the strongest comparison on broad text.** In the clearest from-scratch experiment, a model with internal states beat an ordinary model when both saw one billion targets. The internal states cost about 3.78× as much training matrix work per target. Given the same work, the ordinary model could train on more text and won on both broad prediction and arithmetic answer loss. [Original comparison and uncertainty](https://github.com/dario-fumarola/autonomous-compute-trainium/blob/15d52172a8773c905d9951641ea9c0cbc3451fd5/results/latent_exposure_20261030/work-recovery-v1/REPORT.md)

| Development model | Training targets | Broad text loss ↓ | Arithmetic answer loss ↓ | Correct answer prefixes |
|---|---:|---:|---:|---:|
| Ordinary | 1.000B | 2.832195 | 2.309341 | 4/259 |
| Generated internal state | 1.000B | 2.801700 | 2.255895 | 5/259 |
| Trained constant pauses | 1.000B | 2.901458 | 2.307166 | 4/259 |
| Ordinary, matched training work | 3.783B | **2.691525** | **2.167148** | 4/259 |

Loss is negative log likelihood (NLL), measured in nats. Lower values mean the model assigned more probability to the observed text. The answer counts show how far these models still were from reliable arithmetic. This table comes from one development seed. Work was matched by estimated matrix operations. Elapsed time also includes compilation, communication and host contention.

## Read the experiments in order

| Study | What happened |
|---|---|
| [1. Written calculations and supervised learning](docs/experiments.md#written-calculations-and-supervised-learning) | Generated calculations improved arithmetic answers. A selector lost on ordinary natural text, and training on a computation auxiliary lost to ordinary training in both source orders. A simple mixture of math and English improved both domains. |
| [2. Early versus late pauses](docs/experiments.md#early-versus-late-pauses) | The 0.6B pilot stopped during hardware qualification. A separate, completed 150M natural-text study favored ordinary training across three seeds. |
| [3. Repeated layers and learned workspaces](docs/experiments.md#repeated-layers-and-learned-workspaces) | Extra passes could help a trained model use depth. Stronger ordinary controls erased the proposed efficiency advantage. A pretrained writer also lost to ordinary continuation. |
| [4. Persistent internal states](docs/experiments.md#persistent-internal-states) | The states carried predictive information. Their benefit depended on seed, exposure and the compute budget. The last recorded four-fit experiment has no published quality outcome in these snapshots. |

## Try it

A standard clone includes the aggregate scores and a small working implementation of the latent model. Recalculate the headline differences with Python 3 and no dependencies:

```sh
git clone https://github.com/dario-fumarola/trainium-research.git
cd trainium-research
python3 results.py
```

For the complete training, evaluation and test code from all three projects, run `git submodule update --init` in the clone. Each source resolves to the exact revision used here. The [code guide](docs/code.md) maps research questions to implementations, CPU tests, model downloads and hardware requirements. The [source inventory](docs/sources.md) identifies the history and evidence behind the handoff.
