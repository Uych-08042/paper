"""CPU checks; optional --upstream tests execute pinned official fusion/scoring sources."""
import argparse
import ast
import contextlib
import copy
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import stage45_detection_common as c

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--upstream", type=Path)
_args, _remaining = _parser.parse_known_args()
UPSTREAM = _args.upstream.resolve() if _args.upstream else None
try:
    import torch
except ImportError:
    torch = None


def dataset():
    images, annotations = [], []
    for i in range(1, 5):
        images.append(dict(id=i, width=1024, height=1024, file_name=str(i)+".jpg", video_id=i//3, frame_index=i))
        annotations.append(dict(id=i, image_id=i, category_id=2, bbox=[100, 100, 8, 8], area=64, iscrowd=0, track_id=1))
    return dict(info={}, images=images, annotations=annotations,
                categories=[dict(id=i, name=n) for i, n in enumerate(c.NAMES)])


class IntegrityTests(unittest.TestCase):
    def test_seal_receipt_resume_and_tamper(self):
        with tempfile.TemporaryDirectory(prefix="stage45-test-", dir=Path(__file__).parent) as folder:
            p = Path(folder) / "result.json"
            meta = dict(arm="signals", video_id="3")
            c.atomic_json(p, dict(frames=[]))
            self.assertIsNone(c.committed(p, meta))
            c.commit_result(p, dict(meta=meta, frames=[1]), meta)
            self.assertEqual(c.committed(p, meta)["frames"], [1])
            with p.open("a") as f:
                f.write(" ")
            with self.assertRaisesRegex(ValueError, "result changed"):
                c.committed(p, meta)

    def test_prepare_excludes_checking_groups_and_freezes_sources(self):
        with tempfile.TemporaryDirectory(prefix="stage45-test-", dir=Path(__file__).parent) as folder:
            root = Path(folder)
            data = root / "data"
            (data / "annotations").mkdir(parents=True)
            weights = root / "vid.pt"
            weights.write_bytes(b"fixture weights")
            categories = dataset()["categories"]
            fit, checking = [], []
            for split, offset, rows in (("fit", 0, fit), ("checking", 100, checking)):
                for v in range(offset, offset+8):
                    for frame in range(8):
                        name = "images/v{}/{}.jpg".format(v, frame)
                        p = data / name
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes((split+name).encode())
                        rows.append(dict(id=v*10+frame, file_name=name, width=1024, height=1024,
                                         video_id=v, frame_index=frame))
            for name, rows in (("train", fit+checking), ("val_dev", checking)):
                c.atomic_json(data / "annotations" / (name+".json"), dict(images=rows, annotations=[], categories=categories))
            with patch.object(c, "verify_upstream", return_value=str(root)), patch.object(c, "WEIGHT_SHA256", c.digest(weights)):
                plan = c.prepare(data, root, weights, root / "output")
                fit_ids = {v["video_id"] for v in plan["videos"]["fit"]}
                check_ids = {v["video_id"] for v in plan["videos"]["checking"]}
                self.assertFalse(fit_ids & check_ids)
                for step in range(128):
                    video, frames = c.schedule(plan, step)
                    self.assertIn(video["video_id"], fit_ids)
                    self.assertEqual(len(frames), 8)
                c.load_plan(root / "output")
                weights.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    c.load_plan(root / "output")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                c.safe_image(data, "../vid.pt")


def official_fusion():
    """Execute selected unmodified AST definitions, avoiding unavailable CUDA imports."""
    from torch import nn
    from torch.nn import functional as F
    namespace = dict(torch=torch, nn=nn, F=F, math=math, warnings=warnings, os=os,
                     xavier_uniform_=nn.init.xavier_uniform_, constant_=nn.init.constant_)
    files = [("ultralytics/nn/modules/conv.py", {"Conv", "autopad"}),
             ("ultralytics/nn/modules/ops_dcnv3/modules/dcnv3.py",
              {"to_channels_first", "to_channels_last", "build_norm_layer", "build_act_layer", "_is_power_of_2"}),
             ("ultralytics/nn/modules/ops_dcnv3/functions/dcnv3_func.py",
              {"_get_reference_points", "_generate_dilation_grids", "dcnv3_core_pytorch"}),
             ("ultralytics/nn/modules/block.py", {"DepthSeparableConv3D", "FDCN", "MSTFv1"})]
    for relative, names in files:
        path = UPSTREAM / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        body = [n for n in tree.body if getattr(n, "name", None) in names]
        if {n.name for n in body} != names:
            raise ValueError("Missing official source definitions")
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["MSTFv1"]


@unittest.skipUnless(torch is not None, "CPU torch is optional in the SGLA workspace")
class GateTests(unittest.TestCase):
    def test_signal_sources_and_matched_capacity(self):
        from stage45_temporal_gate import ResidualGates, feature_maps
        torch.set_num_threads(1)
        torch.manual_seed(4)
        x = torch.randn(1, 8, 6, 6)
        kwargs = dict(current=x, residual=x*.2, encoder=x, previous=x,
                      boxes=[], age=0, image_size=1024, age_cap=6, floor=.25)
        base, signal = feature_maps(**kwargs)
        b2, s2 = feature_maps(**dict(kwargs, boxes=[[100,100,400,400,.8,2]], age=9, previous=-x))
        self.assertTrue(torch.equal(base, b2))
        self.assertTrue(torch.equal(signal[:, :3], s2[:, :3]))
        self.assertGreater(float((signal[:, 3:]-s2[:, 3:]).abs().sum()), 0.)
        a = ResidualGates()
        self.assertEqual(sum(p.numel() for p in a.parameters()), 387)
        self.assertTrue(torch.equal(a(0, signal), torch.ones_like(signal[:, :1])))

    @unittest.skipUnless(UPSTREAM, "Pass --upstream for actual official fusion checks")
    def test_official_fusion_identity_gradients_memory_and_resume(self):
        from stage45_temporal_gate import ResidualGates, GateHooks
        from stage45_detection_runtime import tensor_digest
        from stage45_detection_experiment import atomic_torch, load_checkpoint
        torch.set_num_threads(1)
        torch.manual_seed(5)
        cls = official_fusion()
        modules = [cls([8,8,8], mode="conv3d", width=.5).eval().requires_grad_(False) for _ in range(3)]
        snapshots = [tensor_digest(m.state_dict()) for m in modules]
        frames = [(torch.randn(1,8,6,6), torch.randn(1,8,6,6)) for _ in range(4)]
        def rollout(hooks=None):
            state, values = [None]*3, []
            for i, inputs in enumerate(frames):
                if hooks:
                    hooks.begin([], i)
                outputs = [m([*inputs, h]) for m, h in zip(modules, state)]
                values.append([o[0] for o in outputs])
                state = [o[1].detach() for o in outputs]
            return values
        with contextlib.redirect_stdout(io.StringIO()):
            with torch.no_grad():
                native = rollout()
            for arm in ("capacity", "signals"):
                torch.manual_seed(c.DESIGN["seed"])
                gates = ResidualGates()
                hooks = GateHooks(modules, gates, arm, c.DESIGN)
                with torch.no_grad():
                    initial = rollout(hooks)
                for expected, actual in zip(native, initial):
                    for x, y in zip(expected, actual):
                        self.assertTrue(torch.equal(x, y))
                optimizer = torch.optim.AdamW(gates.parameters(), lr=.001, weight_decay=0.)
                def update():
                    optimizer.zero_grad(set_to_none=True)
                    value = rollout(hooks)
                    sum(v.square().mean() for frame in value[1:] for v in frame).backward()
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in gates.parameters()), 0)
                    optimizer.step()
                update()
                first = rollout(hooks)[0]
                for x, y in zip(native[0], first):
                    self.assertTrue(torch.equal(x, y))
                with tempfile.TemporaryDirectory(prefix="stage45-test-", dir=Path(__file__).parent) as folder:
                    p = Path(folder)/"step0001.pt"
                    stub = type("R", (), {"torch": torch, "identity": {"device": "cpu"}})()
                    payload = dict(plan="p", arm=arm, runtime=stub.identity, step=1,
                                   gates=gates.state_dict(), optimizer=optimizer.state_dict(), history=[dict(step=1)])
                    atomic_torch(torch, p, payload)
                    c.atomic_json(p.with_suffix(".receipt.json"), c.seal(dict(schema=c.SCHEMA,
                        sha256=c.digest(p), plan="p", arm=arm, step=1)))
                    update()
                    expected = tensor_digest(gates.state_dict())
                    old = load_checkpoint(stub, p, {"fingerprint":"p"}, arm)
                    gates.load_state_dict(old["gates"])
                    optimizer.load_state_dict(old["optimizer"])
                    update()
                    self.assertEqual(tensor_digest(gates.state_dict()), expected)
                    with p.open("ab") as f:
                        f.write(b"tampered")
                    with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                        load_checkpoint(stub, p, {"fingerprint":"p"}, arm)
                hooks.close()
        self.assertEqual([tensor_digest(m.state_dict()) for m in modules], snapshots)


@unittest.skipUnless(UPSTREAM, "Pass --upstream for actual official AP tests")
class OfficialAPTests(unittest.TestCase):
    def test_full_three_arm_summary_requires_complete_committed_videos(self):
        import stage45_detection_summary as summary
        scorer = summary.official_evaluator(UPSTREAM)
        gt = dataset()
        with tempfile.TemporaryDirectory(prefix="stage45-test-", dir=Path(__file__).parent) as folder:
            root = Path(folder)
            annotation = root/"val_dev.json"
            c.atomic_json(annotation, gt)
            videos = [dict(video_id=str(v), frames=[dict(i, frame_id=i["frame_index"]) for i in gt["images"] if i["video_id"] == v]) for v in (0,1)]
            plan = dict(fingerprint="p", upstream=str(UPSTREAM), videos=dict(checking=videos),
                        annotations=dict(checking=dict(path=str(annotation))))
            bindings = {}
            for arm in ("capacity", "signals"):
                path = root/"training"/arm/"step0128.pt"
                path.parent.mkdir(parents=True)
                path.write_bytes(arm.encode())
                bindings[arm] = c.digest(path)
            run = c.seal(dict(schema=c.SCHEMA, plan="p", models=bindings))
            c.atomic_json(root/"evaluation/run.json", run)
            for arm in c.ARMS:
                for video in videos:
                    meta = dict(run=run["fingerprint"], arm=arm, video_id=video["video_id"])
                    rows = [dict(image_id=f["id"], frame_id=f["frame_id"], head_calls=1,
                                 boxes=[[100,100,108,108,.9,2]], ms=1., gains=[]) for f in video["frames"]]
                    c.commit_result(c.result_path(root, arm, video), dict(meta=meta, frames=rows), meta)
            with patch.object(c, "load_plan", return_value=plan), patch.object(summary, "official_evaluator", return_value=scorer), patch.dict(c.DESIGN, bootstrap_samples=8):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary.summarize(root)
                decision = c.read_sealed(root/"analysis/stage45_decision.json")
                self.assertAlmostEqual(decision["primary"]["delta"], 0., places=12)
                self.assertEqual(decision["full_trajectories"], 6)
                self.assertFalse(decision["external_generalization_established"])
                path = c.result_path(root, "signals", videos[0])
                path.unlink()
                with self.assertRaisesRegex(ValueError, "missing behind receipt"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        summary.summarize(root)

    def test_cached_video_bootstrap_equals_cloned_dataset_and_empty_predictions(self):
        from stage45_detection_summary import official_evaluator, build_evaluator, cached_sample, ap, miss_diagnostics
        scorer = official_evaluator(UPSTREAM)
        gt = dataset()
        # Ignore slot retained, with a real ignored annotation and a matching ignored detection.
        gt["annotations"].append(dict(id=9, image_id=1, category_id=4, bbox=[200,200,20,20], area=400, iscrowd=1, ignore=1))
        preds = [dict(image_id=i, category_id=2, bbox=[100,100,8,8], score=.9) for i in (1,2,3)]
        preds.append(dict(image_id=1, category_id=4, bbox=[200,200,20,20], score=.95))
        ev = build_evaluator(gt, preds, scorer)
        chosen = [3,4,3,4,1,2]
        expected = cached_sample(ev, chosen)
        clone = dict(info={}, categories=gt["categories"], images=[], annotations=[])
        clone_predictions = []
        for new, old in enumerate(chosen, 1):
            clone["images"].append(dict(id=new, width=1024, height=1024))
            for a in gt["annotations"]:
                if a["image_id"] == old:
                    clone["annotations"].append(dict(a, id=len(clone["annotations"])+1, image_id=new))
            clone_predictions.extend(dict(p, image_id=new) for p in preds if p["image_id"] == old)
        naive = build_evaluator(clone, clone_predictions, scorer)
        self.assertAlmostEqual(expected["AP"], ap(naive), places=12)
        self.assertAlmostEqual(expected["AP_es"], ap(naive, "0-12"), places=12)
        empty = build_evaluator(gt, [], scorer)
        self.assertEqual(ap(empty), 0.)
        self.assertEqual(ap(empty, "0-12"), 0.)
        video = dict(video_id="7", frames=[dict(id=i, frame_id=i) for i in range(1, 7)])
        annotations = {i:[dict(category_id=2, bbox=[100,100,8,8], track_id=1)] for i in range(1,7)}
        rows = [dict(boxes=[], ms=1., gains=[]) for _ in video["frames"]]
        d = miss_diagnostics(video, rows, annotations)
        self.assertEqual((d["recall"], d["longest_miss_frames"], d["miss_frames_in_runs_ge5"]), (0., 6, 6))


if __name__ == "__main__":
    unittest.main(argv=[__file__, *_remaining])
