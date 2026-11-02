"""Cross-layer size policy for VAE checkpoint archives.

The secure file reader and NPZ codec must enforce the same
maximum state archive size.
"""

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_artifact as artifact,
    variational_autoencoder_checkpoint_npz as codec,
)


def test_artifact_reader_matches_npz_archive_limit():
    assert (
        artifact.MAX_STATE_FILE_BYTES
        == codec.MAX_ARCHIVE_BYTES
    )


def test_candidate_archive_limit_is_64_mib():
    assert codec.MAX_ARCHIVE_BYTES == 64 * 1024 * 1024


def test_archive_limit_exceeds_observed_model_archives():
    # The largest observed complete state archive for the
    # current four-dataset VAE architecture is ~13.109 MiB.
    assert codec.MAX_ARCHIVE_BYTES > 14 * 1024 * 1024
