"""Synthetic protocol, success-classification and publication safeguards."""

from pathlib import Path
import tempfile
import unittest

from wrist_v3.protocol import write_new, read, settings
from wrist_v3.metrics import ap_delta
from wrist_v3.reports import outcome


def result_fixture(passed=(False, False), fp=(101, 121), recovered=(1, 1)):
    methods = {'V3_standard': fp[0], 'V3_rarity': fp[1], 'C': 105, 'F': 125}
    return {'scientific_gates': {name: {'strict_success_both_tracks': value} for name, value in zip(('standard', 'rarity'), passed)},
        'operating_points': {name: {'conf0.25_iou0.5': {'total_fp': value}} for name, value in methods.items()},
        'changes': {name: {'recovery': {'0.5': {'classes': {'fracture': {'recovered': value}}}}}
                    for name, value in zip(('V3_standard-A', 'V3_rarity-D'), recovered)}}


class V3ProtocolTests(unittest.TestCase):
    def test_configuration_keeps_both_checkpoints_and_old_limits(self):
        cfg, legacy = settings()
        self.assertEqual(set(cfg['checkpoints']), {'standard', 'rarity'})
        self.assertEqual((cfg['maximum_model_calls'], cfg['maximum_local_views']), (3, 2))
        self.assertEqual(cfg['ownership_iou'], legacy['matching']['global_local_iou_min'])
        self.assertEqual(cfg['evaluation']['fp_ratio_max'], 1.1)

    def test_success_requires_both_checkpoints_on_both_tracks(self):
        self.assertEqual(outcome(result_fixture((True, True))), 'SUCCESS')
        self.assertEqual(outcome(result_fixture((True, False))), 'PARTIAL IMPROVEMENT')

    def test_lower_FP_without_recovery_is_not_declared_partial_success(self):
        self.assertEqual(outcome(result_fixture(recovered=(0, 0))), 'NO DEMONSTRATED IMPROVEMENT')
        self.assertEqual(outcome(result_fixture()), 'PARTIAL IMPROVEMENT')

    def test_missing_class_delta_does_not_invent_zero_AP(self):
        self.assertEqual(ap_delta({'fracture': {'ap50': .4}, 'bonelesion': {'ap50': .5}},
                                  {'fracture': {'ap50': .3}}), {'fracture': {'ap50': .4-.3}})

    def test_write_new_preserves_previous_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'receipt.json'
            write_new(path, {'passed': True})
            with self.assertRaises(FileExistsError):
                write_new(path, {'passed': False})
            self.assertEqual(read(path), {'passed': True})


if __name__ == '__main__':
    unittest.main()
