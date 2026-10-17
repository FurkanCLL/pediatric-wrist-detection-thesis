"""Matched full-image sampler prototype; never connected to the official trainer."""

import math
import numpy as np

from wrist_design.sampling import DesignPopulation


def offers(capacity, available):
    a, b = available
    return np.array([min(capacity, a), min(capacity, b), min(capacity, a+b)], dtype=int)


def feasible(total_offers, quotas):
    return quotas[0] <= total_offers[0] and quotas[1] <= total_offers[1] and sum(quotas) <= total_offers[2]


# Two stratum quotas share patient capacities; preserve residual feasibility at each choice.
def select_extras(tickets, weights, patients, strata, capacities, quotas, rng):
    tickets = np.sort(np.asarray(tickets, dtype=int))
    capacity = np.asarray(capacities, dtype=int).copy()
    need = np.asarray(quotas, dtype=int).copy()
    if len(np.unique(tickets)) != len(tickets) or np.any(weights[tickets] <= 0) or np.any(capacity < 0) or np.any(need < 0):
        raise ValueError("Invalid bounded image tickets or patient capacities")
    available = np.zeros((len(capacity), 2), dtype=int)
    np.add.at(available, (patients[tickets], strata[tickets]), 1)
    total = sum((offers(c, a) for c, a in zip(capacity, available)), np.zeros(3, dtype=int))
    if not feasible(total, need):
        raise ValueError("Matched bonus quotas are infeasible under the patient/image caps")
    # One random key per source image keeps common tickets paired when eligibility differs.
    priority = -np.log(np.maximum(rng.random(len(weights))[tickets], np.finfo(float).tiny))/weights[tickets]
    selected = []
    for ticket in tickets[np.argsort(priority, kind="stable")]:
        if not need.any():
            break
        p, s = patients[ticket], strata[ticket]
        before = offers(capacity[p], available[p])
        available[p, s] -= 1
        after_skip = offers(capacity[p], available[p])
        total += after_skip-before
        if capacity[p] and need[s]:
            after_accept = offers(capacity[p]-1, available[p])
            remaining = need.copy(); remaining[s] -= 1
            if feasible(total+after_accept-after_skip, remaining):
                selected.append(ticket)
                capacity[p] -= 1
                need = remaining
                total += after_accept-after_skip
        if not feasible(total, need):
            raise ValueError("Weighted ordering lost a feasible matched completion")
    if need.any():
        raise ValueError("Matched bonus quotas were not filled")
    return np.asarray(selected, dtype=np.int32)


class MatchedPopulation(DesignPopulation):
    def __init__(self, rows, specification, difficulty, proposal):
        super().__init__(rows, specification)
        self.difficulty = np.asarray(difficulty, dtype=float)
        if self.difficulty.shape != (self.n,) or not np.isfinite(self.difficulty).all() or np.any((self.difficulty < 0) | (self.difficulty > 1)):
            raise ValueError("Difficulty must be bounded and aligned to sorted training rows")
        self.proposal = proposal
        if not 0 <= proposal["difficulty_weight"] < 1 or not 0 <= proposal["replacement_fraction"] <= .05 or not .95 <= proposal["unique_stratum_minimum"] <= 1:
            raise ValueError("Prototype weights/budget/coverage exceed the bounded proposal")
        self.policy = dict(specification["candidates"]["B"])
        positive = self.present[:, 0]
        self.requested = sum(math.floor((positive == s).sum()*proposal["replacement_fraction"]) for s in (False, True))
        self.policy["bonus_fraction"] = self.requested/self.n
        # Every original view from the small lesion cohort stays exactly once, not only lesion-positive views.
        lesion = np.isin(self.patient, self.lesion_patients)
        self.protected = self.protected | lesion
        self.difficulty = self.difficulty.copy()
        self.difficulty[lesion] = 0
        self.rarity = self.weights(self.policy)[1]/self.policy["weight_cap"]

    def paired(self, seed, epoch):
        reference, reference_flags, accounting = self.sample(self.policy, seed, epoch)
        base = reference[~reference_flags]
        if len(np.unique(base)) != len(base):
            raise ValueError("Shared coverage skeleton contains duplicate source images")
        alpha = self.proposal["difficulty_weight"]
        weights = (1-alpha)*self.rarity+alpha*self.difficulty
        strata = self.present[:, 0].astype(int)
        quotas = np.bincount(strata[reference[reference_flags]], minlength=2)
        capacities = np.minimum(self.policy["maximum_patient_extra_absolute"],
            np.ceil(self.patient_size*self.policy["maximum_patient_extra_fraction"]).astype(int))
        capacities[self.lesion_patients] = 0
        plans = []
        for priority in (self.rarity, weights):
            tickets = np.sort(base[priority[base] > 0])
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 617]))
            extra = select_extras(tickets, priority, self.patient, strata, capacities, quotas, rng)
            draws = np.concatenate([base, extra])
            flags = np.r_[np.zeros(len(base), dtype=bool), np.ones(len(extra), dtype=bool)]
            order = rng.permutation(self.n)
            plans.append((draws[order], flags[order]))
        (rare, rare_flags), (guided, guided_flags) = plans
        self.check(rare); self.check(guided)
        if not np.array_equal(np.unique(rare), np.unique(guided)) or int(guided_flags.sum()) != int(rare_flags.sum()):
            raise ValueError("Rarity/error comparison changed coverage or realized intervention budget")
        return rare, rare_flags, guided, guided_flags, accounting

    def check(self, draws):
        draws = np.asarray(draws)
        repeat = np.bincount(draws, minlength=self.n)
        unique = np.flatnonzero(repeat)
        if len(draws) != self.n or repeat.max() > 2 or np.unique(self.patient[unique]).size != len(self.patient_size):
            raise ValueError("Draw/image/patient guardrail failed")
        for positive in (False, True):
            status = self.present[:, 0] == positive
            if status[unique].sum() < math.ceil(self.proposal["unique_stratum_minimum"]*status.sum()):
                raise ValueError("Per-epoch unique fracture-stratum coverage failed")
            if status[draws].sum() != status.sum():
                raise ValueError("Fracture-stratum draw quota changed")
        extras = np.bincount(self.patient, weights=np.maximum(repeat-1, 0), minlength=len(self.patient_size))
        cap = np.minimum(self.policy["maximum_patient_extra_absolute"],
            np.ceil(self.patient_size*self.policy["maximum_patient_extra_fraction"]).astype(int))
        if np.any(extras > cap) or np.any(repeat[np.isin(self.patient, self.lesion_patients)] != 1):
            raise ValueError("Patient repetition or lesion-cohort protection failed")
