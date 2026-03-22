import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.domain_gate import DomainGate


def test_domain_gate_pass(tmp_path):
    centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    np.save(str(tmp_path / "centroid.npy"), centroid)

    gate = DomainGate(
        centroid_path=str(tmp_path / "centroid.npy"),
        similarity_threshold=0.5,
    )

    query_embedding = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    allowed, score = gate.check(query_embedding)
    assert allowed is True
    assert score > 0.5


def test_domain_gate_reject(tmp_path):
    centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    np.save(str(tmp_path / "centroid.npy"), centroid)

    gate = DomainGate(
        centroid_path=str(tmp_path / "centroid.npy"),
        similarity_threshold=0.5,
    )

    query_embedding = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    allowed, score = gate.check(query_embedding)
    assert allowed is False
    assert score < 0.5
