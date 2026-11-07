"""Check hardware binding and same-device comparisons after a source drift."""
import contextlib
import csv
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import stage75_l20_replay as replay
import test_stage75_ppo_failure as fixture

c = fixture.c
original = fixture.experiment


class DeviceBinding(unittest.TestCase):
    def test_l20_uuid_and_only_hardware_difference(self):
        runtime = dict(device="NVIDIA L40", torch="sealed-torch", cuda="sealed-cuda")
        identity = dict(tracker=runtime, model_state_sha256="frozen", gpu_uuid="fixture-l40")
        plan = dict(runtime=runtime, runtime_identity=identity)
        devices = [SimpleNamespace(name="NVIDIA L20", uuid="fixture-l20"),
                   SimpleNamespace(name="NVIDIA L40", uuid="fixture-l40")]
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: len(devices),
                               get_device_name=lambda index: devices[index].name,
                               get_device_properties=devices.__getitem__)
        torch = SimpleNamespace(cuda=cuda)
        uid = replay.l20_uuid(plan, 0, torch)
        run = replay.runtime_record(dict(fingerprint="fixture"), dict(plan=plan), uid)
        self.assertEqual(run["runtime"], dict(identity, tracker=dict(runtime, device="NVIDIA L20"), gpu_uuid=uid))
        self.assertEqual({row["field"] for row in run["declared_runtime_differences"]},
                         {"runtime.gpu_uuid", "runtime.tracker.device"})
        self.assertEqual(identity["tracker"]["device"], "NVIDIA L40")
        for index in (1, 2):
            with self.assertRaisesRegex(ValueError, "needs NVIDIA L20"):
                replay.l20_uuid(plan, index, torch)
        with self.assertRaisesRegex(ValueError, "physical L20 UUID"):
            replay.runtime_record(dict(fingerprint="fixture"), dict(plan=plan), "unavailable")


class DeviceMatchedReplay(fixture.RealReplay):
    def fixture_source(self, root, original_source, *args):
        plan = original_source["plan"]
        plan["runtime"]["device"] = "NVIDIA L40"
        plan["runtime_identity"].update(tracker=plan["runtime"], gpu_uuid="fixture-l40")
        return super().fixture_source(root, original_source, *args)

    def check_replay(self, root74, observer, digest):
        source = c.load_source(root74)
        plan74 = source["plan"]
        entry = plan74["entries"][0]
        sequence = entry["sequence"]
        units = {(unit["policy"], unit["seed"]): unit for unit in plan74["units"] if unit["sequence"] == sequence}
        q0 = c.read_trace(source, units["q0", 0])
        truth = np.asarray(entry["ground_truth"])[1:]
        reference = c.failure_events(c.tracking_iou(q0["boxes"], truth))[0]["start"] + 1
        ppo = c.read_trace(source, units["ppo", 2171])
        frame = next(frame for frame in range(2, len(ppo["frame_id"]))
                     if c.source_common.action_mask(ppo["candidate_boxes"][frame - 1]).sum() > 1)
        base = c.case_spec("active", sequence, 2171, reference, "fixture", "first_failure", "q0", 0)
        specs = (dict(base, offset=frame - reference),
                 dict(base, case_id="last", offset=len(ppo["frame_id"]) - reference))
        root = root74.parent / "stage75_l20"
        frozen = digest(observer.rt.net)
        saved_tail = original.collect_tail

        def drift(_module, _inputs, output):
            # Change observed numeric outputs with an unchanged real CENTER state_dict.
            return dict(output, size_map=output["size_map"] * 1.01)

        def interrupted(*args, **kwargs):
            if args[5] != args[4]["recorded_action"]:
                raise RuntimeError("interrupted alternative")
            return saved_tail(*args, **kwargs)

        hook = observer.rt.net.register_forward_hook(drift, prepend=True)
        try:
            with patch.object(c, "CASES", specs), fixture.writer_lock(root), contextlib.redirect_stdout(io.StringIO()):
                plan, cached = replay.prepare(root, source)
                run = replay.runtime_record(plan, source, "fixture-l20")
                with patch.object(observer, "capture", wraps=observer.capture) as calls:
                    with patch.object(original, "collect_tail", side_effect=interrupted):
                        with self.assertRaisesRegex(RuntimeError, "interrupted alternative"):
                            replay.replay(root, plan, source, run, cached, observer)
                    completed = calls.call_count
                    self.assertEqual(completed, 4 * plan["cases"][0]["horizon"])
                    calls.reset_mock()
                    report = replay.replay(root, plan, source, run, cached, observer)
                    self.assertEqual(calls.call_count, report["search_calls"] - completed)
                self.assertEqual(report["training_updates"], 0)
                with (root / "analysis/stage75_l20_device_bridge.csv").open(newline="", encoding="utf-8") as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(row["exact_source_window_match"], "False")
                self.assertGreater(float(row["anchor_max_candidate_error_px"]), 0.)
                self.assertIn("boxes", row["differing_fields"].split("|"))
                with (root / "analysis/stage75_l20_branches.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                controls = [row for row in rows if row["is_l20_ppo_branch"] == "True"]
                self.assertEqual(len(controls), 2)
                self.assertTrue(all(float(row["mean_gain"]) == 0. for row in controls))
                self.assertEqual(controls[-1]["following_mean_gain"], "")

                case = plan["cases"][0]
                meta = dict(schema=replay.SCHEMA, plan_fingerprint=plan["fingerprint"],
                            run_fingerprint=run["fingerprint"], case_id=case["case_id"], kind="anchor")
                anchor = replay.cache.read_cache(root / "collection/active/anchor.npz", meta)
                pack = original.e74.learned_pack(source["models"], case["unit"], source["prior"])
                history = original.e63.history_at(ppo, entry["initial_box"], frame, plan74)
                # Simulate a device-dependent duplicate mask and verify baseline selection uses the local mask.
                inputs = {name: anchor[name][0].copy() for name in ("features", "candidate_boxes", "candidate_parents")}
                inputs["candidate_boxes"][:] = inputs["candidate_boxes"][0]
                decision = original.e74.choose(inputs, history, case["unit"], plan74, source["prior"], pack)
                alias = dict(anchor, **{name: np.asarray([value]) for name, value in decision.items()})
                local = replay.audit_anchor(alias, ppo, case, entry, source, pack)
                self.assertEqual(local["legal_actions"], [0])
                self.assertEqual(local["recorded_action"], 0)
                self.assertNotEqual(local["recorded_action"], case["recorded_action"])
                bad = dict(anchor, crop_sha256=np.full_like(anchor["crop_sha256"], "changed"))
                with self.assertRaisesRegex(AssertionError, "Shared anchor input changed"):
                    replay.audit_anchor(bad, ppo, case, entry, source, pack)

                with patch.dict("sys.modules", torch=None, cv2=None):
                    self.assertEqual(report, replay.replay(root, plan, source, run, cached))
                c.verify(source["bindings"])
                self.assertEqual(digest(observer.rt.net), frozen)
        finally:
            hook.remove()


if __name__ == "__main__":
    unittest.main()
