"""A bounded hybrid sampler prototype, independent of the official training loader."""

import numpy as np

from .common import arrays


class DesignPopulation:
    def __init__(self, rows, specification):
        self.rows = sorted(rows, key=lambda r: r["filestem"])
        if len({r["filestem"] for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate training image identity")
        self.specification = specification
        self.counts, self.patient, patients = arrays(self.rows)
        self.present = self.counts > 0
        self.n = len(rows)
        self.patient_size = np.bincount(self.patient, minlength=len(patients))
        self.lesion_patients = np.unique(self.patient[self.present[:, 5]])
        self.patient_support = np.array([len(np.unique(self.patient[self.present[:, c]])) for c in range(8)])
        # Rare-image protection and status-specific anchors preserve broad independent coverage.
        self.protected = self.present[:, specification["protected_classes"]].any(axis=1) | ~self.present.any(axis=1)
        self.groups = [[np.flatnonzero((self.patient == p) & (self.present[:, 0] == positive))
                        for p in range(len(patients))] for positive in (False, True)]

    def weights(self, policy):
        reference = self.specification["reference_image_support"]
        class_weights = np.zeros(8)
        for c in self.specification["target_classes"]:
            support = self.patient_support[c]
            if support >= self.specification["independent_patient_floor"] and self.present[:, c].sum():
                reliability = min(1., support/self.specification["patient_reliability_reference"])
                class_weights[c] = min(policy["weight_cap"], np.sqrt(reference/self.present[:, c].sum())*reliability)
        # Saturation by maximum prevents co-occurring categories from adding extreme weights.
        image_weights = (self.present*class_weights).max(axis=1)
        image_weights[np.isin(self.patient, self.lesion_patients)] = 0
        return class_weights, image_weights

    def sample(self, policy, seed, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        if policy is None:
            return rng.permutation(self.n), np.zeros(self.n, dtype=bool), {"requested_bonus": 0, "rare_bonus": 0, "coverage_fallback": 0}
        _, weights = self.weights(policy)
        extras = np.zeros(len(self.patient_size), dtype=np.int32)
        cap = np.minimum(policy["maximum_patient_extra_absolute"],
                         np.ceil(self.patient_size*policy["maximum_patient_extra_fraction"]).astype(int))
        cap[self.lesion_patients] = 0
        total_bonus = round(self.n*policy["bonus_fraction"])
        positive_bonus = round(total_bonus*self.present[:, 0].sum()/self.n)
        requested = {False: total_bonus-positive_bonus, True: positive_bonus}
        draws, bonus_flags = [], []
        rare_bonus = fallback = 0
        for positive in (False, True):
            members = np.flatnonzero(self.present[:, 0] == positive)
            selected = set(int(i) for i in members if self.protected[i])
            for group in self.groups[int(positive)]:
                if len(group) and not any(int(i) in selected for i in group):
                    selected.add(int(rng.choice(group)))
            # If a small fixture is too dense for the requested redistribution, protect coverage first.
            base_size = max(len(selected), len(members)-requested[positive])
            remaining = np.array([i for i in members if i not in selected], dtype=int)
            selected.update(map(int, rng.permutation(remaining)[:base_size-len(selected)]))
            base = np.array(sorted(selected), dtype=int)
            needed = len(members)-len(base)
            eligible = base[weights[base] > 0]
            tickets = np.repeat(eligible, policy["maximum_image_draws"]-1)
            # Exponential priorities give a weighted permutation of bounded image tickets.
            priorities = -np.log(np.maximum(rng.random(len(tickets)), np.finfo(float).tiny))/weights[tickets]
            additional = []
            for ticket in tickets[np.argsort(priorities, kind="stable")]:
                patient = self.patient[ticket]
                if len(additional) == needed:
                    break
                if extras[patient] < cap[patient]:
                    additional.append(int(ticket))
                    extras[patient] += 1
            rare_bonus += len(additional)
            recovery = [int(i) for i in rng.permutation(members) if i not in selected][:needed-len(additional)]
            fallback += len(recovery)
            draws.extend([*map(int, base), *additional, *recovery])
            bonus_flags.extend([False]*len(base) + [True]*len(additional) + [False]*len(recovery))
        order = rng.permutation(len(draws))
        draws = np.asarray(draws, dtype=np.int32)[order]
        flags = np.asarray(bonus_flags, dtype=bool)[order]
        if len(draws) != self.n or int(self.present[draws, 0].sum()) != int(self.present[:, 0].sum()):
            raise ValueError("Draw budget or fracture status quota changed")
        return draws, flags, {"requested_bonus": total_bonus, "rare_bonus": rare_bonus, "coverage_fallback": fallback}
