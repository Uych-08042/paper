"""Fresh-group separation and actual PyTorch gradient/rollout/resume tests."""

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from stage4_common import file_hash, write_cache, write_json
from stage27_uav_common import crop_geometry, seal
import stage31_identity_engine as old_engine
from stage31_identity_common import fit_ranker
from stage31_precision_compat import compatibility_record, precision_replay_auditor
from stage34_identity_common import ARMS, DESIGN, SCHEMA, code_hashes, fresh_entries, load_plan, prepare_plan, sample_weights, supervision_masks
import stage34_identity_experiment as experiment
from stage34_identity_runtime import forward_student, losses, make_student, spatial_masks, template_weights
from stage34_identity_summary import compare_rows, decide, summarize
from test_stage31_precision_compat import Float32Observer, contract_fixture
from test_stage27_uav_diagnostic import fake_image


HERE = Path(__file__).resolve().parent
TORCH = importlib.util.find_spec("torch") is not None


class NearObserver(Float32Observer):
    def initial_anchor(self, image, box):
        return np.r_[1., np.zeros(self.identity["embed_dim"] - 1)]

    def observe(self, image, features):
        packet = super().observe(image, features)
        packet["boxes"][0, 0] = 28
        if features:
            expanded = np.zeros((len(packet["descriptors"]), self.identity["embed_dim"]))
            expanded[:, :2] = packet["descriptors"]
            packet["descriptors"] = expanded
            if self.frame % 4 < 2:
                packet["boxes"] = packet["boxes"][::-1].copy()
                packet["descriptors"] = packet["descriptors"][::-1].copy()
        packet["native_box"] = packet["boxes"][0].copy()
        return packet


def source_fixture(root):
    source, contract, reader = contract_fixture(root)
    contract["runtime"] = dict(embed_dim=8, lens_z=16, lens_x=16, parameter="fake")
    contract = seal({k: v for k, v in contract.items() if k != "fingerprint"})
    write_json(source / "manifest.json", contract)
    run = seal(dict(schema=contract["schema"], role="native_fit_candidate_features", contract_fingerprint=contract["fingerprint"]))
    write_json(source / "fit_run.json", run)
    write_json(source / "stage31_precision_compat.json", compatibility_record(contract))
    records, hashes = [], {}
    with precision_replay_auditor():
        for entry in contract["sequences"]:
            if entry["split"] == "fit":
                data = old_engine.run_video(NearObserver(contract["runtime"]), contract, entry, "sgla", fit_collect=True, image_reader=reader)
                old_engine.audit_video(data, contract, entry, "sgla", fit_collect=True)
                path = source / "fit_cache" / (entry["sequence"] + ".npz")
                write_cache(path, data, old_engine.cache_metadata(run, entry, "sgla"))
                records.append(data)
                hashes[path.name] = file_hash(path)
    result = fit_ranker(records)
    static = []
    for data in records:
        f = np.zeros_like(data["features"])
        f[..., [0, 7, 8]] = data["features"][..., [0, 7, 8]]
        static.append(dict(data, features=f))
    sr = fit_ranker(static)
    model = seal(dict(schema=contract["schema"], role="frozen_fit_only_identity_ranker", **result,
        contract_fingerprint=contract["fingerprint"], fit_run_fingerprint=run["fingerprint"], fit_cache_sha256=hashes,
        static_weights=sr["weights"], static_feature_scale=sr["feature_scale"], checking_labels_used=False))
    write_json(source / "model.json", model)
    prior = root / "stage33"
    from stage33_descriptor_probe import dependency_hashes
    prior_plan = seal(dict(schema="stage33_pooled_descriptor_oof_v1", source_contract_fingerprint=contract["fingerprint"],
        role="consumed_development_fit_only_fixed_budget_lovo", code_sha256=dependency_hashes()))
    write_json(prior / "plan.json", prior_plan)
    artifacts = {}
    names = []
    for fold in range(8):
        for seed in (2100, 2101):
            for arm in ("scalar_balanced", "descriptor_fixed", "descriptor_learned"):
                stem = "fold{:02d}_seed{}_{}".format(fold, seed, arm)
                names += ["checkpoints/" + stem + ".npz", "checkpoints/" + stem + ".json", "oof_predictions/" + stem + ".npz"]
    for i, name in enumerate(names):
        p = prior / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(str(i).encode())
        artifacts[p.relative_to(prior).as_posix()] = file_hash(p)
    write_json(prior / "stage33_decision.json", seal(dict(schema=prior_plan["schema"], mode="full", role="development_only_pooled_descriptor_feasibility",
        plan_fingerprint=prior_plan["fingerprint"], checking_labels_loaded=False, artifacts_sha256=artifacts)))
    dataset = Path(contract["root"])
    for index in range(100, 106):
        name = "uav{:07d}_00000_s".format(index)
        p = dataset / "sequences" / name
        p.mkdir(parents=True)
        for i in range(25):
            (p / ("img{:07d}.jpg".format(i + 1))).write_bytes(bytes([i + 1]) * 12)
        (dataset / "annotations" / (name + ".txt")).write_text("\n".join(["40,20,10,10"] * 25))
    # Excluded old checking annotations deliberately disappear before prepare.
    for entry in contract["sequences"]:
        if entry["split"] == "checking":
            (dataset / entry["annotation"]).unlink()
    return source, prior, contract


def torch_runtime(torch):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.GELU(), torch.nn.Linear(32, 8))
            self.norm = torch.nn.LayerNorm(8)
        def forward(self, x):
            return x + self.ffn(self.norm(x))
    class Patch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 1)
        def forward(self, x):
            return self.conv(x).flatten(2).transpose(1, 2)
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.score = torch.nn.Linear(8, 1)
            self.size = torch.nn.Linear(8, 2)
            self.offset = torch.nn.Linear(8, 2)
        def cal_bbox(self, score, size, offset):
            ids = score.flatten(1).argmax(1)
            index = ids[:, None, None].expand(-1, 2, 1)
            wh, off = size.flatten(2).gather(2, index).squeeze(-1), offset.flatten(2).gather(2, index).squeeze(-1)
            return torch.cat(((torch.stack((ids % 4, ids // 4), 1) + off) / 4, wh), 1)
        def forward(self, x):
            s = self.score(x).transpose(1, 2).reshape(-1, 1, 4, 4).sigmoid()
            wh = self.size(x).transpose(1, 2).reshape(-1, 2, 4, 4).sigmoid()
            off = self.offset(x).transpose(1, 2).reshape(-1, 2, 4, 4)
            return dict(score_map=s, size_map=wh, offset_map=off, pred_boxes=self.cal_bbox(s, wh, off)[:, None])
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.blocks = torch.nn.ModuleList(Block() for _ in range(12))
            self.backbone.patch_embed = Patch()
            self.backbone.pos_embed_z = torch.nn.Parameter(torch.randn(1, 16, 8) * .2)
            self.backbone.pos_embed_x = torch.nn.Parameter(torch.randn(1, 16, 8) * .2)
            self.backbone.pos_drop = torch.nn.Identity()
            self.backbone.norm = torch.nn.LayerNorm(8)
            self.backbone.MLP = torch.nn.Linear(32, 6)
            self.backbone.cat_mode = "direct"
            self.box_head = Head()
        def forward_head(self, tokens, unused):
            return self.box_head(tokens[:, -16:])
    class RT:
        def __init__(self):
            self.torch = torch
            self.net = Net().eval().requires_grad_(False)
            self.backbone = self.net.backbone
            self.identity = dict(embed_dim=8, lens_z=16, lens_x=16, parameter="fake")
            self.lens_z, self.lens_x, self.search_side = 16, 16, 4
            self.s1 = SimpleNamespace(combine_tokens=lambda z, x, **kw: torch.cat((z, x), 1), recover_tokens=lambda x, *a, **kw: x)
            self.tracker = SimpleNamespace(params=SimpleNamespace(search_factor=4, search_size=256, template_factor=2, template_size=128),
                                           output_window=torch.ones(1, 1, 4, 4))
        def patch(self, image):
            x = torch.as_tensor(image.copy()).float().permute(2, 0, 1)[None] / 255
            return torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        def initialize(self, image, box):
            self.tracker.z_dict1 = SimpleNamespace(tensors=self.patch(image))
            self.tracker.state = list(box)
        def prepare(self, image):
            _, _, factor = crop_geometry(self.tracker.state, image.shape[:2], 4, 256)
            return self.patch(image), factor
        def prefix(self, search):
            bb = self.backbone
            x = torch.cat((bb.patch_embed(self.tracker.z_dict1.tensors) + bb.pos_embed_z, bb.patch_embed(search) + bb.pos_embed_x), 1)
            for block in bb.blocks[:6]:
                x = block(x)
            return x
        def head(self, tokens):
            return self.net.forward_head(self.backbone.norm(tokens), None)
        def selector(self, tokens):
            return tokens.new_zeros((1, 6))
        def selected_layer(self, values):
            return 9
        def candidate(self, tokens, layer):
            return self.head(self.backbone.blocks[layer - 1](tokens))
        def decode(self, output, factor, image):
            p = self.net.box_head.cal_bbox(output["score_map"], output["size_map"], output["offset_map"])[0].detach().numpy() * 256 / factor
            previous = np.asarray(self.tracker.state)
            center = previous[:2] + previous[2:] / 2 - 256 / factor / 2 + p[:2]
            wh = np.maximum(p[2:], 2)
            xy = np.clip(center - wh / 2, [0, 0], [image.shape[1] - 2, image.shape[0] - 2])
            wh = np.minimum(wh, [image.shape[1], image.shape[0]] - xy)
            return np.r_[xy, wh]
    return RT()


class DataTests(unittest.TestCase):
    def test_interaction_and_initial_guard_use_videos_not_seed_repetitions(self):
        rows = []
        aucs = dict(sgla=60., fast_initial=50., readout_standard=50., readout_contrast=50.1,
                    representation_standard=51., representation_contrast=52.)
        for i in range(6):
            for method, auc in aucs.items():
                seeds = DESIGN["seeds"] if method in ARMS else ["baseline"]
                for seed in seeds:
                    rows.append(dict(sequence=str(i), method=method, seed=seed, auc=auc,
                        low_iou_fraction=0., failure_events=0, instrumented_ms=1.))
        with mock.patch.dict(DESIGN, bootstrap_samples=64):
            averaged, pairs = compare_rows(rows)
            report = decide(rows, averaged, pairs)
            self.assertEqual(report["status"], "PASS_SPATIAL_REPRESENTATION_PILOT")
            self.assertAlmostEqual(report["interaction"]["mean_delta"], .9)
            self.assertTrue(all(r["videos"] == 6 for r in pairs))
            bad = copy.deepcopy(rows)
            next(r for r in bad if r["method"] == "representation_contrast")["failure_events"] = 1
            averaged, pairs = compare_rows(bad)
            self.assertFalse(decide(bad, averaged, pairs)["guard_pass"])

    def test_prepare_excludes_old_groups_and_training_sampling_is_balanced(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            source, prior, contract = source_fixture(root)
            before = {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
            plan = prepare_plan(source, prior, root / "stage34", fake_image, 20, 500)
            self.assertEqual(len(plan["fit"]), 8)
            self.assertEqual(len(plan["checking"]), 6)
            self.assertFalse({e["source_group_proxy"] for e in plan["checking"]} & set(plan["excluded_checking_groups"]))
            weights = sample_weights(plan["samples"])
            hard = np.array([r["cohort"] == "hard" for r in plan["samples"]])
            self.assertAlmostEqual(float(weights[hard].sum()), .5)
            self.assertEqual(plan, load_plan(root / "stage34"))
            self.assertEqual(before, {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()})
            with self.assertRaisesRegex(ValueError, "Insufficient untouched"):
                fresh_entries(Path(contract["root"]), {e["source_group_proxy"] for e in contract["sequences"]}, 7, fake_image, 20, 500)


@unittest.skipUnless(TORCH, "Install/use PyTorch to run actual representation gradient tests")
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def test_zero_gain_identity_real_loss_and_gradient_intervention(self):
        torch = self.torch
        rt = torch_runtime(torch)
        image = np.ones((64, 96, 3), np.uint8)
        rt.initialize(image, [40, 20, 10, 10])
        search, _ = rt.prepare(image)
        entry = dict(initial_box=[40, 20, 10, 10], frames=[dict(height=64, width=96)])
        weights = template_weights(rt, entry)
        initial_outputs = rt.head(rt.prefix(search))
        for arm in ARMS:
            net = make_student(rt, arm)
            out, _ = forward_student(rt, net, search, weights)
            torch.testing.assert_close(out["score_map"], initial_outputs["score_map"], atol=0, rtol=0)
            torch.testing.assert_close(out["pred_boxes"], initial_outputs["pred_boxes"], atol=0, rtol=0)
            self.assertEqual(len(net.backbone.blocks), 6)
            self.assertFalse(hasattr(net.backbone, "MLP"))
            target = torch.tensor([[.25, .25, .25, .25]])
            total, _, contrast, _, counts = losses(rt, net, search, weights, target, arm)
            self.assertTrue(torch.isfinite(total))
            total.backward()
            prefix = list(net.backbone.blocks[4].parameters()) + list(net.backbone.blocks[5].parameters())
            if arm.startswith("representation"):
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in prefix if p.grad is not None), 0)
            else:
                self.assertTrue(all(p.grad is None for p in prefix))
            self.assertTrue(all(c > 0 for c in counts))
        positive, negative = spatial_masks(torch, torch.tensor([[.501, .501, .001, .001]]), 16, 16)
        self.assertEqual(int(positive.sum()), 1)
        self.assertFalse((positive & negative).any())
        centered, _ = spatial_masks(torch, torch.tensor([[.479, .479, .001, .001]]), 16, 16)
        self.assertTrue(bool(centered[0, 8, 8]))  # round(.4795 * 16), same as CENTER.
        for target in ([.25, .25, .25, .25], [.501, .501, .001, .001], [.05, .05, .9, .9]):
            p, n = spatial_masks(torch, torch.tensor([target], dtype=torch.float64), 16, 16)
            np_p, np_n = supervision_masks(target, 16)
            np.testing.assert_array_equal(p[0].numpy(), np_p)
            np.testing.assert_array_equal(n[0].numpy(), np_n)

    def test_matched_train_resume_and_all_fresh_video_evaluations(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            with mock.patch.dict(DESIGN, updates=2, smoke_updates=2, save_every=1, bootstrap_samples=64):
                source, prior, _ = source_fixture(root)
                output = root / "stage34"
                plan = prepare_plan(source, prior, output, fake_image, 20, 500)
                rt = torch_runtime(self.torch)
                # No fresh checking image/annotation should be opened by training.
                fit_names = {e["sequence"] for e in plan["fit"]}
                def fit_reader(root, frame):
                    self.assertIn(Path(frame["file"]).parent.name, fit_names)
                    return fake_image(Path(root) / frame["file"])
                smoke = experiment.train_experiment(rt, plan, output, DESIGN["seeds"][0], True, fit_reader)
                self.assertEqual(smoke["status"], "PASS_SPATIAL_IDENTITY_SMOKE")
                original_save = experiment.save_step
                def interrupt_once(torch, directory, value):
                    path = original_save(torch, directory, value)
                    if value["step"] == 1:
                        raise RuntimeError("injected interruption after committed step")
                    return path
                with mock.patch.object(experiment, "save_step", side_effect=interrupt_once):
                    with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                        experiment.train_experiment(rt, plan, output, DESIGN["seeds"][0], False, fit_reader)
                for seed in DESIGN["seeds"]:
                    decision = experiment.train_experiment(rt, plan, output, seed, False, fit_reader)
                    self.assertEqual(decision["status"], "COMPLETE_MATCHED_TRAINING")
                reference = experiment.train_arm(rt, plan, root / "uninterrupted_reference", "readout_standard", DESIGN["seeds"][0], False, fit_reader)
                resumed = output / "training" / ("seed{}".format(DESIGN["seeds"][0])) / "readout_standard" / "step000002.pt"
                expected = experiment.load_checkpoint(self.torch, Path(reference["checkpoint"]))["parameters"]
                actual = experiment.load_checkpoint(self.torch, resumed)["parameters"]
                for key in expected:
                    self.torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                checkpoints = {str(p): file_hash(p) for p in (output / "training").rglob("*.pt")}
                experiment.train_experiment(rt, plan, output, DESIGN["seeds"][0], False, fit_reader)
                self.assertEqual(checkpoints, {str(p): file_hash(p) for p in (output / "training").rglob("*.pt")})
                experiment.evaluate(rt, plan, output, lambda root, frame: fake_image(Path(root) / frame["file"]))
                self.assertEqual(len(list((output / "checking").glob("*.npz"))), 84)
                report = summarize(output)
                self.assertEqual(report["videos"], 6)
                self.assertIn(report["status"], ("PASS_SPATIAL_REPRESENTATION_PILOT", "NO_SPATIAL_REPRESENTATION_GAIN_ESTABLISHED"))
                for r in plan["checking"]:
                    for p in (output / "checking").glob(r["sequence"] + "__*.npz"):
                        with np.load(p, allow_pickle=False) as a:
                            self.assertEqual(int(a["main_head_calls"]), len(r["frames"]) - 1)
                corrupt = next((output / "training").rglob("*.pt"))
                corrupt.write_bytes(corrupt.read_bytes() + b"changed checkpoint bytes")
                seed = int(corrupt.parents[1].name.removeprefix("seed"))
                with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                    experiment.train_experiment(rt, plan, output, seed, False, fit_reader)


if __name__ == "__main__":
    unittest.main()
