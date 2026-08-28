from __future__ import annotations

import numpy as np

from atari_d3pm.cli.stage5a import pairwise_agreement


def test_pairwise_agreement_separates_aliases_from_motion_disagreement():
    predictions = [
        np.asarray([0, 2, 4, 3, 5]),
        np.asarray([0, 4, 2, 5, 3]),
    ]
    result = pairwise_agreement(predictions)["0__1"]
    assert result["raw"] == 0.2
    assert result["canonical4"] == 1.0
    assert result["alias_only"] == 0.8
