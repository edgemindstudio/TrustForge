"""Compute-node three-scale MS-SSIM fixture; no primary pool evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(args.repository_root).resolve()
    require(root == Path.cwd().resolve(), "checkout mismatch")

    def check_source():
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
        require(head == args.expected_head, "revision mismatch")
        subprocess.run(["git", "diff", "--quiet"], check=True)
        subprocess.run(["git", "diff", "--cached", "--quiet"], check=True)

    check_source()
    require(not args.output.exists() and not args.output.is_symlink(),
            "output already exists")

    from trustforge.evaluation.paper01_classifier_tensorflow import (
        require_compute_node,
    )
    require_compute_node()

    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    tf.config.threading.set_inter_op_parallelism_threads(1)

    from trustforge.evaluation.paper01_ms_ssim import ThreeScaleMSSSIM

    scorer = ThreeScaleMSSSIM()
    rng = np.random.default_rng(42)
    first = rng.random((3, 40, 40, 3), dtype=np.float32)
    second = rng.random((3, 40, 40, 3), dtype=np.float32)
    first_before, second_before = first.copy(), second.copy()

    observed = scorer.score_pairs(first, second)
    direct = tf.image.ssim_multiscale(
        first, second, max_val=1.0,
        power_factors=(1 / 3, 1 / 3, 1 / 3),
        filter_size=7, filter_sigma=1.5, k1=0.01, k2=0.03,
    ).numpy()
    require(np.array_equal(observed, direct), "direct TensorFlow agreement")
    require(np.array_equal(observed, scorer.score_pairs(first, second)),
            "same-input replay")
    require(np.array_equal(first, first_before)
            and np.array_equal(second, second_before), "input mutation")
    require(np.allclose(scorer.score_pairs(first, first), 1,
                        rtol=0, atol=1e-6), "identity score")

    # Analytic constant-channel reference: contrast terms equal one.
    # TensorFlow combines scale terms per channel, then averages channels.
    red = np.zeros((1, 40, 40, 3), dtype=np.float32)
    green = red.copy()
    red[..., 0] = 1
    green[..., 1] = 1
    rgb_score = float(scorer.score_pairs(red, green)[0])
    c1 = 0.01 ** 2
    analytic_rgb = (2 * (c1 / (1 + c1)) ** (1 / 3) + 1) / 3
    require(np.isclose(rgb_score, analytic_rgb, rtol=1e-4, atol=1e-5),
            "constant RGB analytic reference")

    require(
        all(side >= 7 for side in (40, 20, 10)),
        "three-scale filter geometry",
    )

    rejected = 0
    malformed = [
        first.astype(np.float64),
        first[:, :, :, :1],
        np.zeros((1, 39, 40, 3), dtype=np.float32),
        np.zeros((0, 40, 40, 3), dtype=np.float32),
        np.zeros((17, 40, 40, 3), dtype=np.float32),
        np.full((1, 40, 40, 3), np.nan, dtype=np.float32),
        np.full((1, 40, 40, 3), 1.1, dtype=np.float32),
        np.full((1, 40, 40, 3), -0.1, dtype=np.float32),
    ]
    for bad in malformed:
        try:
            scorer.score_pairs(bad, second[:1])
        except ValueError:
            rejected += 1
        else:
            raise RuntimeError("malformed input accepted")
    try:
        scorer.score_pairs(first, second[:1])
    except ValueError:
        rejected += 1
    else:
        raise RuntimeError("unequal pair counts accepted")

    report = {
        "scope": "THREE_SCALE_MS_SSIM_SYNTHETIC_FIXTURE",
        "source_commit": args.expected_head,
        "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "scorer": scorer.identity,
        "scale_geometry": [40, 20, 10],
        "direct_tensorflow_agreement": True,
        "same_input_replay": True,
        "inputs_unchanged": True,
        "identity_score_passed": True,
        "constant_rgb_reference_passed": True,
        "constant_rgb_observed": rgb_score,
        "constant_rgb_expected": analytic_rgb,
        "invalid_cases_rejected": rejected,
        "fixture_scores": observed.tolist(),
        "fixture_scores_sha256":
            hashlib.sha256(observed.tobytes(order="C")).hexdigest(),
        "primary_pool_evaluated": False,
        "complete_evaluation": False,
        "full_benchmark_go": False,
    }
    check_source()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("THREE_SCALE_MS_SSIM_FIXTURE=PASS")


if __name__ == "__main__":
    main()
