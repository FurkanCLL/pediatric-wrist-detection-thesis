"""Synthetic regression checks for validation-only inference and error accounting."""

import json
import unittest
from unittest.mock import patch
import numpy as np

from wrist_eval.common import CLASSES, ROOT, config, identity, image_path
from wrist_eval.metrics import (ap_summary, ious, match, native_stats, paired_recovery,
                               size_bin, validate_predictions, remove_padding_only)
from wrist_eval.sliced import resize_and_pad, tile_dimensions


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.gt=np.array([[0,0,10,10,0],[20,0,30,10,3]],dtype=np.float32)
        self.pred=np.array([[0,0,10,10,.9,0],[0,0,10,10,.8,0],[20,0,30,10,.7,4]],dtype=np.float32)

    def test_one_to_one_duplicate(self):
        g,p,_=match(self.gt,self.pred)
        self.assertEqual(g.tolist(),[0,-1]); self.assertEqual(p.tolist(),[0,-1,-1])
        self.assertEqual(sum(p>=0),sum(g>=0))

    def test_class_agnostic_confusion(self):
        g,_,_=match(self.gt,self.pred,class_aware=False)
        self.assertEqual(g.tolist(),[0,2])

    def test_confidence_and_iou(self):
        g,_,_=match(self.gt,self.pred,confidence=.95)
        self.assertTrue(np.all(g<0))
        shifted=np.array([[4,0,14,10,.9,0]],dtype=np.float32)
        self.assertLess(ious(self.gt[:1,:4],shifted[:,:4])[0,0],.5)
        self.assertEqual(match(self.gt,shifted)[0][0],-1)

    def test_native_ap_integration(self):
        perfect=np.array([[0,0,10,10,.9,0],[20,0,30,10,.8,3]],dtype=np.float32)
        stats=native_stats(self.gt,perfect)
        self.assertTrue(stats["tp"].all())
        result=ap_summary([stats])
        self.assertAlmostEqual(result["map50_95"],.995)
        duplicate=native_stats(self.gt,self.pred)
        self.assertEqual(duplicate["tp"][:,0].tolist(),[True,False,False])

    def test_recovery_without_double_count(self):
        b=np.array([[20,0,30,10,.9,3],[20,0,30,10,.8,3]],dtype=np.float32)
        pair=paired_recovery(self.gt,self.pred,b)
        self.assertEqual(pair["recovered"].tolist(),[False,True])
        self.assertEqual(pair["lost"].tolist(),[True,False])

    def test_invalid_predictions(self):
        for box in ([0,0,0,1,.2,0],[0,0,1,1,float("nan"),0],[0,0,1,1,.2,8],[-1,0,1,1,.2,0],[0,0,11,1,.2,0]):
            with self.assertRaises(ValueError):validate_predictions([box],10,10)
        self.assertEqual(validate_predictions([],10,10).shape,(0,6))

    def test_stratum_edges(self):
        self.assertEqual([size_bin(x) for x in (0,15.9,16,32,64,128)], [0,0,1,2,3,4])

    def test_padding_only_export_is_counted(self):
        result,count=remove_padding_only([[0,0,10,0,.0014,0],[1,1,2,2,.8,0]])
        self.assertEqual(count,1);self.assertEqual(len(result),1)

    def test_validation_access_only(self):
        for split in ("train","test"):
            with self.assertRaises(ValueError):image_path({"split":split,"filestem":"synthetic"})
        with self.assertRaises(ValueError):image_path({"split":"val","filestem":"../test/synthetic"})

    def test_config_identity_and_mapping(self):
        cfg=config()
        self.assertEqual(identity(cfg),identity(json.loads(json.dumps(cfg))))
        self.assertEqual(len(CLASSES),8)
        self.assertEqual(cfg["conf"],.001);self.assertTrue(cfg["multi_label"])
        from wrist_train.protocol import load_protocol
        self.assertEqual(load_protocol()[0]["training"]["epochs"],80)

    def test_aspect_padding_immutable(self):
        image=np.full((1100,417,3),45,np.uint8)
        before=image.copy()
        padded,ratio=resize_and_pad(image)
        self.assertTrue(np.array_equal(image,before))
        self.assertEqual(padded.shape[0]%32,0);self.assertEqual(padded.shape[1]%32,0)
        self.assertLess(abs(ratio[0][0]-ratio[0][1]),1/417)
        self.assertEqual(tile_dimensions(1100,417,.5),(550,209))

    def test_sahi_coverage_coordinates_merge(self):
        from sahi.slicing import get_slice_bboxes
        from sahi.prediction import ObjectPrediction
        from sahi.postprocess.combine import NMSPostprocess
        for h,w in ((1100,417),(400,800),(50,50)):
            sh,sw=tile_dimensions(h,w,.5)
            boxes=get_slice_bboxes(h,w,slice_height=sh,slice_width=sw,overlap_height_ratio=.2,overlap_width_ratio=.2,auto_slice_resolution=False)
            coverage=np.zeros((h,w),bool)
            for x1,y1,x2,y2 in boxes:coverage[y1:y2,x1:x2]=True
            self.assertTrue(coverage.all())
        a=ObjectPrediction(bbox=[1,2,11,12],category_id=0,score=.9,shift_amount=[20,30],full_shape=[100,100]).get_shifted_object_prediction()
        self.assertEqual(a.bbox.to_xyxy(),[21,32,31,42])
        b=ObjectPrediction(bbox=[21,32,31,42],category_id=0,score=.8)
        c=ObjectPrediction(bbox=[21,32,31,42],category_id=3,score=.7)
        result=NMSPostprocess(match_threshold=.7,match_metric="IOU",class_agnostic=False)([a,b,c])
        self.assertEqual(len(result),2)
        self.assertEqual({r.category.id for r in result},{0,3})

    def test_patient_cluster_resampling(self):
        from wrist_eval.bootstrap import patient_groups,resample_indices
        groups=patient_groups([{"patient_id":"a"},{"patient_id":"b"},{"patient_id":"a"}])
        indices=resample_indices(groups,np.random.default_rng(42))
        self.assertEqual(indices.count(0),indices.count(2))
        self.assertEqual(indices,resample_indices(groups,np.random.default_rng(42)))

    def test_descriptive_accounting(self):
        from wrist_eval.analyze import error_types
        g,p,_,reasons,fp=error_types(self.gt,self.pred)
        self.assertEqual(reasons.tolist(),["TP","classification"])
        self.assertEqual(fp.tolist(),["TP","duplicate","classification"])
        self.assertEqual(int(sum(g>=0)),int(sum(p>=0)))

    def test_pilot_is_outcome_independent(self):
        from wrist_eval.feasibility import pilot_rows
        rows=[{"filestem":f"synthetic{i}","patient_id":str(i),"width":500+i,"height":900,
               "boxes":[{"class_id":j,"xyxy":[1,1,20+j,40]} for j in range(8)]} for i in range(30)]
        first=pilot_rows(rows)
        self.assertEqual([r["filestem"] for r in first],[r["filestem"] for r in pilot_rows(list(reversed(rows)))])
        self.assertEqual(len(first),24)


if __name__=="__main__":unittest.main()
