# CIC AR recipe-report contract

**Decision.** PASS for the bounded checkpoint-backed contract diagnostic.

Execution revision `dee4c60fd68aee63e0927547f76e75ebcba01cad`.
Recipe SHA-256 `202db292bb18d3e6f04f6f8cc36658884994c3bd9f4f48e2e93e181d348ea478`.
Diagnostic directory `ar-cic-recipe-contract-dee4c60-retry-9hqh5NDN`.

Failed array `311611` ran both tasks on talon08. Both exited with code
`7:0` after 1m 38s during TensorFlow startup. The cause remains unresolved.

Retry array `311628` excluded talon08 and allowed one task at a time.
Both tasks completed on talon16 with exit code `0:0`.

The verification checked all four batch archives, recipe identity,
class-0 conditioning IDs, and sampling seeds 23501–23504.
The finalizer accepted report identities and seeds, then rejected the
four-batch pool at its required 1,250-batch coverage check.
No output directory or final manifest was created.

The verifier printed `CIC_AR_RECIPE_CONTRACT=PASS`.
Full checkpoint-backed pool publication and evaluator integration remain
pending. Full Paper 1 benchmark status remains **NO-GO**.
