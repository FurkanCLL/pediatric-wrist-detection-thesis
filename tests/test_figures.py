"""Geometry and privacy checks for presentation-only exports."""

import unittest
import numpy as np

from wrist_figures.data import prepared_paths, decode_yolo, mask_text, crop_bounds


class FigureTests(unittest.TestCase):
    def test_heldout_and_path_escape_are_rejected(self):
        for row in ({'split': 'test', 'filestem': 'fixture'}, {'split': 'train', 'filestem': '../fixture'}):
            with self.assertRaises(ValueError):
                prepared_paths(row)

    def test_actual_yolo_edge_coordinates_are_restored(self):
        values = decode_yolo('3 0.5 0.25 0.2 0.1\n', 1000, 2000)
        np.testing.assert_allclose(values, [[400, 400, 600, 600, 3]], rtol=0, atol=1e-10)

    def test_malformed_labels_are_rejected(self):
        for value in ('3 0.5 0.5 0.1', '8 0.5 0.5 0.1 0.1', '0 nan .5 .1 .1', '0 .5 .5 -.1 .1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_yolo(value, 100, 100)

    def test_redaction_changes_an_export_copy_only(self):
        pixels = np.full((100, 100), 255, np.uint8)
        masked, regions = mask_text(pixels, [{'class_id': 2, 'xyxy': [10, 20, 30, 40]},
            {'class_id': 0, 'xyxy': [50, 50, 80, 80]}])
        self.assertEqual(regions, [[6, 16, 34, 44]])
        self.assertTrue((pixels == 255).all())
        self.assertTrue((masked[16:44, 6:34] == 0).all())
        self.assertEqual(int(masked[60, 60]), 255)

    def test_text_redaction_clamps_at_image_edges(self):
        image = np.ones((20, 30), np.uint8)
        masked, regions = mask_text(image, [{'class_id': 2, 'xyxy': [0, 0, 30, 20]}])
        self.assertEqual(regions, [[0, 0, 30, 20]])
        self.assertFalse(masked.any())

    def test_display_crops_keep_real_polygon_envelopes(self):
        for box in ([0, 0, 10, 20], [590, 980, 600, 1000], [290, 100, 300, 900]):
            crop = crop_bounds([box], 600, 1000)
            self.assertTrue(0 <= crop[0] <= box[0] < box[2] <= crop[2] <= 600)
            self.assertTrue(0 <= crop[1] <= box[1] < box[3] <= crop[3] <= 1000)


if __name__ == '__main__':
    unittest.main()
