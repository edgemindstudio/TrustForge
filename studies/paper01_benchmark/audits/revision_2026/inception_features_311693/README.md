# Inception RGB feature bounded acceptance

Decision: PASS for synthetic-fixture feature extraction.

- Job 311693: COMPLETED|0:0 on talon07; elapsed 00:00:23.
- Execution revision: c4d4d8f24c8dde9f98918af0582080a48a3690e8.
- Explicit verified no-top ImageNet weights loaded successfully.
- RGB preservation and accepted preprocessing passed fixture checks.
- Features were finite float32 with shape (3,2048).
- Same-input inference replay was exact in the observed environment.
- Inference left model weights unchanged.
- Selected-position iteration preserved order with batches [16,1].
- Tested malformed inputs were rejected.
- Saved report identity and extractor contract were verified.

Weight SHA-256:
9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a

Fixture feature-byte SHA-256:
7933fc376d48138eb0af6c9f01949a7fee01ab10e9b0fe4caf9b3ee8485a258a

Limits: synthetic RGB fixtures only. This does not establish primary
reference-to-feature integration, real-data provenance, cross-environment
bitwise replay, malware-semantic feature validity or complete metrics.

FULL_BENCHMARK_GO=NO
