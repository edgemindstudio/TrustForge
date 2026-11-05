"""Observe unchanged MAF spline arithmetic; never publish checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


class BoundaryObserved(RuntimeError):
    pass


class DiagnosticComplete(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    report_path = Path(args.report)
    if report_path.exists() or report_path.is_symlink():
        raise ValueError("diagnostic report destination already exists")

    import numpy as np
    from trustforge.models.paper01_revision_2026 import (
        masked_autoflow_flow as flow,
        masked_autoflow_spline as spline,
        masked_autoflow_training as training,
        ar_maf_checkpoint_publication as publication,
    )

    context = {"training_step_started": 0, "spline_call_in_step": 0}
    original_forward = flow.spline_forward
    original_train = training.train_step

    def write_report(record):
        document = {
            "purpose": "UNCHANGED_MAF_BOUNDARY_DIAGNOSTIC",
            "source_head_requested": args.expected_head,
            "job_id": os.environ["SLURM_JOB_ID"],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "context": dict(context),
            "checkpoint_publication_allowed": False,
            **record,
        }
        with report_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(
                document, indent=2, sort_keys=True, allow_nan=False
            ) + "\n")
        print("MAF_BOUNDARY_DIAGNOSTIC", json.dumps(
            document, sort_keys=True, allow_nan=False
        ), flush=True)

    def observe_forward(x, raw_widths, raw_controls):
        context["spline_call_in_step"] += 1
        # Call the original implementation with the original tensors.
        y, logdet = original_forward(x, raw_widths, raw_controls)
        values = y.numpy()
        bad = np.argwhere(
            ~np.isfinite(values) | (values < 0) | (values > 1)
        )
        if bad.size:
            widths, heights, x_knots, y_knots = spline.spline_parameters(
                raw_widths, raw_controls,
            )
            arrays = {
                "input": x.numpy(),
                "raw_widths": raw_widths.numpy(),
                "raw_slope_controls": raw_controls.numpy(),
                "widths": widths.numpy(),
                "heights": heights.numpy(),
                "x_knots": x_knots.numpy(),
                "y_knots": y_knots.numpy(),
            }
            examples = []
            for row, column in bad[:8]:
                entry = {"batch_row": int(row), "coordinate": int(column)}
                for name, array in arrays.items():
                    value = array[row, column]
                    entry[name] = (
                        float(value) if np.ndim(value) == 0 else value.tolist()
                    )
                value = values[row, column]
                entry["output"] = float(value) if np.isfinite(value) else str(value)
                entry["output_float32_bits"] = int(
                    np.asarray(value, dtype=np.float32).view(np.uint32).item()
                )
                examples.append(entry)
            write_report({
                "outcome": "FIRST_OUT_OF_RANGE_SPLINE_OUTPUT",
                "violation_count": int(len(bad)),
                "examples": examples,
            })
            raise BoundaryObserved("first invalid spline output captured")
        return y, logdet

    def observe_train(*positional, **keywords):
        context["training_step_started"] += 1
        context["spline_call_in_step"] = 0
        return original_train(*positional, **keywords)

    def block_publication(*positional, **keywords):
        write_report({
            "outcome": "TRAINING_REACHED_PUBLICATION_WITHOUT_CAPTURED_VIOLATION",
        })
        raise DiagnosticComplete("publication intentionally blocked")

    flow.spline_forward = observe_forward
    training.train_step = observe_train
    publication.publish_checkpoint_bundle = block_publication

    root = Path(args.repository_root).resolve()
    sys.argv = [
        str(root / "scripts/check_paper01_ar_maf_cic_checkpoint_pilot.py"),
        "--family", "maf", "--phase", "train",
        "--repository-root", str(root),
        "--expected-head", args.expected_head,
        "--dataset-config", args.dataset_config,
        "--config-sha256", args.config_sha256,
        "--publication-parent", str(report_path.parent),
        "--generation-name", "diagnostic_never_published",
    ]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except (BoundaryObserved, DiagnosticComplete) as exc:
        print("DIAGNOSTIC_STOP:", str(exc), flush=True)


if __name__ == "__main__":
    main()
