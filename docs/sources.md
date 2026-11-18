# Where the studies came from

The study directories contain regular files from the October 16–30, 2026 research. Each imported file retains its original contents and executable mode. The one omitted file is a conversation excerpt in the 1.7B project's context directory. The August–September EMA snapshot predates this collection.

| Bundled study | Original repository | Source revision |
|---|---|---|
| [01-useful-thoughts](../studies/01-useful-thoughts/) | useful-thoughts-trainium | `d07d89e0348cf1acd76acd1173a90dcd11facf3e` |
| [02-pause-pilot](../studies/02-pause-pilot/) | pause-pretraining-trainium | `a7d6a14c30b12c5c6a920f04c1c1b5121065afa2` |
| [03-autonomous-compute](../studies/03-autonomous-compute/) | autonomous-compute-trainium | `15d52172a8773c905d9951641ea9c0cbc3451fd5` |

## Work outside the main branches

Six useful-thoughts development tips had commits outside the final main history. The files below preserve each branch's changes from its common ancestor with main. They record the implementation at that stage. Use the complete study directories for terminal results.

| Branch changes | Common ancestor | Branch tip |
|---|---|---|
| [candidate-utility-v30](../studies/development-branches/candidate-utility-v30.patch) | `35161e2389aa44465efdcb59636c8b27d6b2a9b8` | `a84c77b89fd469e40536353b8744856136984e03` |
| [causal-decoder-batching-v1](../studies/development-branches/causal-decoder-batching-v1.patch) | `2eb3c1cf09d14604cc9b5dd0f97562656e95b6a5` | `b25560a9618970a43afab349edc785d772a02604` |
| [document-evidence-v12](../studies/development-branches/document-evidence-v12.patch) | `9f05ae051425d1d70883503d3a85bdb086a722dd` | `7c5c26414945e3f1c7c937bfac98a13762d4c7b0` |
| [general-sft-v5](../studies/development-branches/general-sft-v5.patch) | `731a9cea8b6223828da9642271e3d61ba0c652b4` | `9e3f4b298d4a4870e3be4844bc9c34de2f4dfef5` |
| [real-prefix-partial-v9](../studies/development-branches/real-prefix-partial-v9.patch) | `ed0700372941080f9cec22a54ac60f36549c56da` | `9d57df721e81fa88b3f87ab2766184b9b38ae819` |
| [variable-depth-eval-v11](../studies/development-branches/variable-depth-eval-v11.patch) | `9f05ae051425d1d70883503d3a85bdb086a722dd` | `11512db569ce62dd844c9bb530149018413826c3` |

## Original score releases

The [replay directory](../replay/) contains the unpacked score releases. Each included SHA256SUMS manifest covers the replay program, numerical inputs, recorded readout and supporting files. Release archive hashes are recorded in the original publication receipts. [Baseline receipt](../studies/01-useful-thoughts/validation/uniform_lm_baseline_public_release_20261022.json) · [Scaffold receipt](../studies/01-useful-thoughts/validation/scaffold_learning_public_release_20261022.json) · [Mixture receipt](../studies/01-useful-thoughts/validation/uniform_mixture_public_release_20261022.json)

The research materials retain their original notices. Models and datasets have their own terms. [Research notices](../studies/01-useful-thoughts/NOTICE.md) · [Third-party code](../studies/03-autonomous-compute/third_party/) · [Base model license](../tools/restore-model/BASE_MODEL_LICENSE.txt)
