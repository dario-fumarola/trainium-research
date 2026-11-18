# Matched computation supervision: ordinary training wins in both orders

The prespecified supervised-learning criterion **fails**. Adding verified computation supervision makes direct prediction **worse than the same-data ordinary-training control in both source orders**, by 0.000944 and 0.000908 nats per token, with negative confidence intervals. It improves on format-only supervision, but also loses to the independently qualified ordinary-training reference. All six trained models improve over untouched B0. Ordinary continued pretraining works here; this auxiliary computation objective does not improve it.

This archive contains the original native per-token scores, fixed document identities, sealed readout, and original training histories where applicable. It excludes source text, source token IDs and model weights.

Run `python replay.py evidence` in this directory with NumPy installed. This reconstructs the original numerical outcomes, including every failure; it does not rerun the backbones or verify omitted source text.

Reviewed report: https://github.com/dario-fumarola/useful-thoughts-trainium/blob/d8db165fef202c846631e43a5518d5c0334dce07/validation/scaffold_learning_complete_20261022/REPORT.md

Every file except SHA256SUMS is bound by that manifest.
