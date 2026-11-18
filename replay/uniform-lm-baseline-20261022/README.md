# Uniform next-token baseline evidence

Original native token scores, all four 128-update histories, terminal inventory and frozen DEV choice. No source text, source token IDs, credentials or model weights are included.

Requires Python and NumPy. Run:

    python replay_uniform_lm.py evidence

This independently reconstructs six document-paired comparisons, the DEV selection and the fresh confirmation decision. It does not rerun models, verify omitted source text or provide an external replication. The confirmation gain is 0.0144812062 nats/token [0.0139032777, 0.0150581179] over B0 on 4,096 FineMath windows. Final 16-token gain is separately 0.0327166462 [0.0135460824, 0.0522745055] nats/span.

Scientific numerical source: 4f9166b944fcc3a7e38df6f32bd45b396d5047ef. Full report and fixed protocol are in the repository. Actual assigned cost including unsuccessful qualification reservations is 16.6072758632 chip-hours; this is not whole-node billing.
