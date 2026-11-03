"""Regression for mixed fixed/event anchor requests and immutable cache recovery."""
import copy
import csv
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import stage67_summary_repair as repair
import stage67_veto_common as c
import stage67_veto_summary as original


class ExportTests(unittest.TestCase):
    def requests(self):
        return [dict(s,status="supported" if i<7 else "no_changed_veto_in_declared_window",
                     resolved_frame=i+2 if i<7 else None) for i,s in enumerate(c.CONFIG)]

    def test_all_eight_request_shapes_optional_fixed_frame_and_no_mutation(self):
        rows=self.requests();before=copy.deepcopy(rows)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path=Path(temp)/"stage67_anchors.csv"
            with self.assertRaisesRegex(ValueError,"frame"):repair.ORIGINAL_PUBLISH(path,rows)
            self.assertFalse(path.exists())  # Old export failed before any destination write.
            repair.publish_table(path,rows)
            with path.open(encoding="utf8",newline="") as stream:out=list(csv.DictReader(stream))
            self.assertEqual(list(out[0]),list(repair.ANCHOR_FIELDS));self.assertEqual(len(out),8)
            self.assertTrue(all(r["frame"]=="" for r in out[:7]))
            self.assertEqual(out[7]["frame"],"664");self.assertEqual(out[7]["resolved_frame"],"")
            self.assertEqual(out[7]["status"],"no_changed_veto_in_declared_window")
            payload=path.read_bytes();repair.publish_table(path,rows);self.assertEqual(path.read_bytes(),payload)
            changed=copy.deepcopy(rows);changed[-1]["resolved_frame"]=660
            with self.assertRaisesRegex(ValueError,"differs"):repair.publish_table(path,changed)
            self.assertEqual(path.read_bytes(),payload)
        self.assertEqual(rows,before)

    def test_nonanchor_tables_keep_exact_bytes_and_reject_unknown_fields(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            old=Path(temp)/"old.csv";new=Path(temp)/"stage67_case_summary.csv"
            rows=[dict(sequence="a",value=.2),dict(sequence="b",value=None)]
            repair.ORIGINAL_PUBLISH(old,rows);repair.publish_table(new,rows)
            self.assertEqual(new.read_bytes(),old.read_bytes())
        rows=self.requests();rows[0]["typo"]=1
        with self.assertRaises(ValueError):repair.anchor_rows(rows)
        with self.assertRaises(ValueError):repair.anchor_rows([])

    def test_failed_source_validation_is_not_bypassed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp,patch.object(repair,"ORIGINAL_SUMMARIZE",side_effect=ValueError("frozen source changed")):
            with self.assertRaisesRegex(ValueError,"frozen source changed"):repair.summarize(temp)
            self.assertEqual(list(Path(temp).iterdir()),[])
        self.assertIs(original.publish_table,repair.ORIGINAL_PUBLISH)


try:
    import torch
    import cv2
except ImportError:torch=cv2=None


@unittest.skipUnless(torch is not None and cv2 is not None,"Real CPU Torch/OpenCV required; skips are not validation")
class Workflow(unittest.TestCase):
    def test_real_frozen_branch_chain_repaired_without_source_changes_or_inference(self):
        from test_stage67_veto import Workflow as Previous
        from stage4_common import file_hash
        saved_resolve=c.resolve;completed=[]

        def mixed_requests(source):
            # Keep the real four-frame fixture's one supported branch point and add
            # the actual production fixed-frame request shape as unsupported.
            fixed=dict(case_id="fixed_control",sequence=c.CONFIG[0]["sequence"],
                       selector="last_veto_before_fixed_frame",frame=664)
            with patch.object(c,"CONFIG",c.CONFIG+(fixed,)):
                return saved_resolve(source)

        def repaired(root):
            root=Path(root)
            protected=[root/"manifest.json",root/"run.json",root/"collection/run.json",root/"smoke/stage67_smoke.json"]
            protected.extend((root/"collection/traces").glob("*"))
            before={str(p):file_hash(p) for p in protected}
            result=repair.summarize(root)
            self.assertTrue(all(file_hash(p)==h for p,h in before.items()))
            receipt=c.prior.sealed(root/"analysis/stage67_summary_repair.json",repair.SCHEMA)
            self.assertEqual(receipt["new_search_calls"],0);self.assertEqual(receipt["new_image_reads"],0)
            self.assertFalse(receipt["source_hash_checks_relaxed"])
            self.assertEqual(receipt["anchor_rows"],2)
            self.assertEqual(receipt["table_sha256"],result["table_sha256"])
            self.assertEqual(receipt["original_code_sha256"],c.sealed(root/"manifest.json")["code_sha256"])
            with (root/"analysis/stage67_anchors.csv").open(encoding="utf8",newline="") as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(rows[-1]["frame"],"664");self.assertEqual(rows[-1]["resolved_frame"],"")
            completed.append(result)
            return result

        with patch.object(c,"resolve",side_effect=mixed_requests),patch.object(original,"summarize",side_effect=repaired):
            Previous().test_real_source_diagonal_gt_isolation_smoke_reuse_resume_and_immutability()
        self.assertGreaterEqual(len(completed),2)


if __name__=="__main__":unittest.main()
