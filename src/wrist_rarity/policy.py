"""Reuse the sealed Phase 4A rarity comparator without any mined difficulty input."""

import json
import numpy as np

from wrist_design.common import training_records, arrays
from wrist_mining.sampling import MatchedPopulation
from wrist_train.common import ROOT
from wrist_train.protocol import canonical_hash


class RarityPlans:
    def __init__(self, rows=None):
        self.rows = training_records() if rows is None else rows
        specification = json.loads((ROOT/"configs/class_aware_candidates.json").read_text())
        proposal = json.loads((ROOT/"configs/error_guided_sampler_proposed.json").read_text())
        if canonical_hash(specification) != proposal["class_aware_candidates_canonical_sha256"]:
            raise ValueError("Historical rarity specification changed")
        # The first paired result depends only on rarity, never on this zero placeholder.
        self.population = MatchedPopulation(self.rows, specification, np.zeros(len(self.rows)), proposal)

    def plan(self, seed, epoch):
        if seed not in (42, 1042, 2042) or not isinstance(epoch, int) or not 0 <= epoch < 80:
            raise ValueError("Unexpected seed or zero-based epoch")
        draws = self.population.paired(seed, epoch)[0]
        self.validate(draws)
        return draws

    def validate(self, draws):
        draws = np.asarray(draws)
        if draws.shape != (14061,) or draws.dtype.kind not in "iu" or np.any(draws < 0) or np.any(draws >= 14061):
            raise ValueError("Invalid rarity-only plan shape or indices")
        self.population.check(draws)
        repeats = np.bincount(draws, minlength=14061)
        if int(np.maximum(repeats-1, 0).sum()) != 625:
            raise ValueError("Approved 625 realized replacements changed")
        return {"draws": 14061, "unique_images": int((repeats > 0).sum()), "replacements": 625,
                "patients": 4264, "fracture_draws": 9307, "no_fracture_box_draws": 4754,
                "lesion_cohort_views_once": 66, "plan_sha256": plan_hash(draws)}


def plan_hash(draws):
    import hashlib
    return hashlib.sha256(np.asarray(draws, dtype="<i4").tobytes()).hexdigest()
