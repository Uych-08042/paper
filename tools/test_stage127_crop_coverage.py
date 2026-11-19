"""Check cached crop geometry, border handling and independence from output quality."""
import unittest

import numpy as np

import stage127_crop_coverage as stage
from stage27_uav_common import crop_geometry, tracking_iou


class CropCoverage(unittest.TestCase):
    def inspect(self, gt, parent, output):
        settings = dict(search_factor=2., search_size=256)
        hw = (100,100)
        parents = np.repeat([parent],4,axis=0)
        predictions = np.repeat([output],4,axis=0)
        crop = crop_geometry(parent,hw,2.,256)[0]
        data = dict(frame_id=[1], alternate_used=[True], selected=[0], image_hw=[hw],
                    candidate_boxes=[predictions], candidate_parents=[parents],
                    crop=np.repeat(crop[None,None,:],4,axis=1))
        entry = dict(frames=[{},dict(height=100,width=100)], ground_truth=[gt,gt])
        recorded = dict(recorded_action="0",candidate_ious="|".join(map(str,tracking_iou(predictions,gt))))
        point = dict(sequence="fixture",source_seed=2312,frame=1,
                     stage124_predictions={str(stage.trigger.SEEDS[0]):recorded})
        return stage.inspect(data,entry,point,settings)

    def test_full_coverage_does_not_require_a_good_output(self):
        point, crops = self.inspect([40,40,10,10],[40,40,10,10],[10,10,10,10])
        self.assertEqual(point["max_coverage_visible_gt"],1.)
        self.assertEqual(point["bucket"],"adequate_coverage")
        self.assertFalse(point["good_output"])
        self.assertEqual(crops[0]["input_target_width"],128.)
        self.assertTrue(point["center_in_any_crop"])

    def test_separates_missing_the_crop_from_missing_the_image(self):
        for gt, bucket in (([40,40,10,10],"zero_crop_overlap"),([102,10,10,10],"target_outside_image")):
            point,_ = self.inspect(gt,[1,1,5,5],[10,10,10,10])
            self.assertEqual(point["bucket"],bucket)
        self.assertIsNone(point["max_coverage_visible_gt"])

    def test_normalizes_only_to_the_target_area_inside_the_frame(self):
        point, _ = self.inspect([-5,40,10,10],[0,40,20,20],[25,25,10,10])
        self.assertEqual(point["gt_image_fraction"],.5)
        self.assertEqual(point["max_coverage_full_gt"],.5)
        self.assertEqual(point["max_coverage_visible_gt"],1.)

    def test_partial_coverage_thresholds(self):
        self.assertEqual(stage.classify(1.,.09),"under_10_percent")
        self.assertEqual(stage.classify(1.,.10),"partial_coverage")
        self.assertEqual(stage.classify(1.,.90),"adequate_coverage")


if __name__=="__main__":
    unittest.main()
