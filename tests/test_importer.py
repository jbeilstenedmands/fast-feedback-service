"""
Tests for the importer
"""

import os
import re
import subprocess
from pathlib import Path

import pytest


def test_importer(dials_data, tmp_path):
    importer_path: str | Path | None = os.getenv("IMPORTER")
    assert importer_path
    d = dials_data("thaumatin_i03_rotation", pathlib=True)
    proc = subprocess.run(
        [
            importer_path,
            d / "thau_2_1.nxs",
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr
    assert (tmp_path / "imported.expt").exists()