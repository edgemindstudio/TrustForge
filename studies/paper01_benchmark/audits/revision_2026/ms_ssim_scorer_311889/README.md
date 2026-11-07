# Three-scale MS-SSIM scorer bounded acceptance

Decision: PASS for synthetic-fixture scorer execution.

- Job 311889: COMPLETED|0:0 on talon07; elapsed 00:01:05.
- Execution revision: 5cd3cb933bd007955c8b913fce319f5b45c00a59.
- TensorFlow 2.20.0; selected ms_ssim_3scale_40x40_v1 settings.
- Canonical RGB inputs used three scales with geometry 40,20,10.
- Direct TensorFlow agreement and same-input replay passed.
- Inputs remained unchanged; identical-image scores passed.
- Constant RGB analytic reference passed within declared tolerance:
  observed 0.36427220702171326; expected 0.3642762274953255.
- Nine malformed-input/pair-count cases were rejected.
- Saved report contract and score-byte identity were verified.

The direct-call check validates binding to TensorFlow; it is not an independent
implementation of the complete multiscale algorithm. The constant RGB check
provides a separate analytic case.

Persisted-pair consumption, all-class aggregation, failed-pair propagation and
complete-pool integration remain open. No primary synthetic pool was evaluated.

Stage 5 remains IN_PROGRESS.
FULL_BENCHMARK_GO=NO
