"""Focused checks for presentation-only data handling and redaction."""

import csv
import tempfile
from pathlib import Path
import unittest

import numpy as np

from wrist_final_figures.data import training_csv, load, METHODS
from wrist_final_figures.qualitative import mask_text, crop_box, clear_of_text
from wrist_final_figures.plots import plt, outside_canvas_text


class FinalFigureTests(unittest.TestCase):
    def test_layout_ignores_tick_artists_outside_the_displayed_range(self):
        fig, ax = plt.subplots(layout='constrained')
        ax.set_xticks([0, 2, 4]); ax.set_xlim(0, 3.5)
        self.assertEqual(outside_canvas_text(fig), [])
        plt.close(fig)

    def test_layout_detects_actual_text_outside_the_canvas(self):
        fig, _ = plt.subplots(layout='constrained')
        fig.text(-.1, .5, 'Outside')
        self.assertIn('Outside', outside_canvas_text(fig))
        plt.close(fig)

    def test_training_logs_require_complete_unique_recorded_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'history.csv'
            with p.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream); writer.writerow(['epoch', 'loss'])
                writer.writerows([[i, 1/i] for i in range(1, 81)])
            self.assertEqual(len(training_csv(p)), 80)
            p.write_text('epoch,loss\n1,1\n1,2\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                training_csv(p)

    def test_nonfinite_training_values_are_not_invented_or_interpolated(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'history.csv'
            p.write_text('epoch,loss\n'+'\n'.join(f'{i},nan' for i in range(1, 81)), encoding='utf-8')
            with self.assertRaises(ValueError):
                training_csv(p)

    def test_redaction_copies_pixels_and_clips_mask_at_image_bounds(self):
        image = np.full((20, 30), 200, np.uint8)
        masked = mask_text(image, [[0, 0, 5, 5]], padding=2)
        self.assertTrue(np.all(masked[:7, :7] == 0))
        self.assertTrue(np.all(image == 200)); self.assertEqual(masked[19, 29], 200)

    def test_display_crop_preserves_target_and_image_bounds(self):
        for target in ([0, 0, 10, 10], [90, 180, 100, 200], [40, 70, 60, 100]):
            x1, y1, x2, y2 = crop_box(target, 100, 200)
            self.assertTrue(0 <= x1 <= target[0] < target[2] <= x2 <= 100)
            self.assertTrue(0 <= y1 <= target[1] < target[3] <= y2 <= 200)

    def test_target_cannot_be_hidden_by_annotated_text_mask(self):
        row = {'boxes': [{'class_id': 2, 'xyxy': [0, 0, 10, 10]}]}
        self.assertFalse(clear_of_text([5, 5, 15, 15], row))
        self.assertTrue(clear_of_text([30, 30, 40, 40], row))

    def test_sources_use_final_V3_and_both_complete_histories(self):
        data = load()
        self.assertEqual([v['id'] for v in data['matrix']['conditions']], METHODS)
        self.assertEqual(data['matrix']['conditions'][2]['inference'], 'cg_alr_v3')
        self.assertEqual(data['matrix']['conditions'][5]['inference'], 'cg_alr_v3')
        self.assertEqual([v['best_epoch'] for v in data['histories'].values()], [22, 26])
        self.assertTrue(all(len(v['rows']) == 80 for v in data['histories'].values()))


if __name__ == '__main__':
    unittest.main()
