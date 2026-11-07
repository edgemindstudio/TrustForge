# Inception no-top weight identity acceptance

Decision: PASS for cached weight-artifact identity only.

Job 311685 completed on talon05 with exit 0:0 in 11 seconds.
Execution revision: 99a98663508061334fd1c1658fc4c173bc2f9c9e.

The 87910968-byte cached artifact matched the installed Keras no-top
ImageNet checksum bcbd6486424b2319ff4ef7d526e38f63.

Weight SHA-256:
9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a

Environment: Keras 3.11.3; TensorFlow 2.20.0.
The installed Inception implementation source hash is retained in report.json.

This verifies agreement with the installed package's artifact specification.
It does not establish model loading, feature preprocessing, RGB handling,
inference behavior, class-balanced feature selection or metric completeness.
Future consumers must verify the artifact they actually load against the
recorded SHA-256. The artifact remains outside Git.

FULL_BENCHMARK_GO=NO
