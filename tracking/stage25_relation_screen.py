"""Audit Stage-25 branches, then run one frozen fit-video grouped-OOF state probe."""

import argparse
from pathlib import Path

import numpy as np

from stage4_common import file_hash, write_cache, write_csv, write_json
from stage8_immediate_gate import grouped_folds
from stage18_representation_train import atomic_torch_save
from stage23_branch_ppo import single_writer
from stage23_development import read_sealed, sealed
from stage25_relation_common import (ANCHORS, BOOTSTRAPS, BOOTSTRAP_SEED, CODE_FILES, FOLDS,
    FOLD_SEED, MODEL_SEEDS, RELATION_DIM, SCHEMA, build_probe, cache_path, discounted_values,
    load_cache, load_contract)


EPOCHS, BATCH = 40, 64


def verify_code(contract):
    for name, digest in contract["code_sha256"].items():
        if Path(name).name != name or file_hash(Path(__file__).parent / name) != digest:
            raise ValueError("Collection/probe code changed after freeze: " + name)
    if not set(CODE_FILES) <= set(contract["code_sha256"]):
        raise ValueError("Incomplete Stage-25 code freeze")


def load_data(directory):
    directory = Path(directory).expanduser().resolve()
    if (directory / ".stage23_writer.lock").exists():
        raise RuntimeError("Collection writer exists; wait for it to exit")
    contract = load_contract(directory)
    verify_code(contract)
    records, missing, hashes = [], [], {"stage25_contract.json": file_hash(directory / "stage25_contract.json")}
    for name in contract["fit"]:
        path = cache_path(directory, name)
        if not path.exists():
            missing.append(name)
            continue
        before = file_hash(path)
        data = load_cache(path, contract, name)
        if file_hash(path) != before:
            raise RuntimeError("Cache changed while reading")
        hashes[path.relative_to(directory).as_posix()] = before
        records.append((name, data))
    return contract, records, missing, hashes


def dataset(records):
    raw, relation, values, post_values, names, anchors = [], [], [], [], [], []
    for name, data in records:
        q, eligible = discounted_values(data["rewards"], data["valid"])
        raw.append(data["raw"][eligible])
        relation.append(data["relation"][eligible])
        values.append(q[eligible])
        post_values.append(discounted_values(data["rewards"], data["valid"], post_anchor=True)[0][eligible])
        names.extend([name] * int(eligible.sum()))
        anchors.extend(data["anchor"][eligible].tolist())
    if not names:
        raise ValueError("No eligible H20 anchors")
    return dict(raw=np.concatenate(raw), relation=np.concatenate(relation), q=np.concatenate(values), post_q=np.concatenate(post_values),
                names=np.asarray(names), anchor=np.asarray(anchors))


def video_weights(names):
    unique, counts = np.unique(names, return_counts=True)
    lookup = dict(zip(unique, counts))
    weights = np.array([1 / lookup[n] for n in names], float)
    return weights / weights.mean()


def standardizer(values, weights):
    mean = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=weights))
    return mean.astype(np.float32), np.maximum(scale, 1e-3).astype(np.float32)


def normalized(values, mean, scale):
    return np.clip((values - mean) / scale, -10, 10).astype(np.float32)


def train_predict(torch, data, train, test, variant, seed, dim, device):
    """Fixed epochs; no test-label early stopping, tuning, or best-seed choice."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_probe(torch, dim).to(device)
    weights = video_weights(data["names"][train])
    raw_mean, raw_scale = standardizer(data["raw"][train], weights)
    rel_mean, rel_scale = standardizer(data["relation"][train], weights)
    x = normalized(data["raw"], raw_mean, raw_scale)
    z = normalized(data["relation"], rel_mean, rel_scale)
    if variant == "raw":
        z = np.zeros_like(z)
    elif variant != "relational":
        raise ValueError("Unknown state variant")
    q = data["q"][train].mean(1)
    centered = q - q.mean(1, keepdims=True)
    target_scale = max(float(np.sqrt(np.average((centered ** 2).mean(1), weights=weights))), 1e-5)
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
    x, z = tensor(x), tensor(z)
    y, w = tensor(centered / target_scale), tensor(weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(EPOCHS):
        for order in np.array_split(rng.permutation(len(train)), (len(train) + BATCH - 1) // BATCH):
            prediction = model(x[train[order]], z[train[order]])
            loss = (((prediction - y[order]) ** 2).mean(1) * w[order]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite state-probe training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite state-probe gradient")
            optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(x[test], z[test]).cpu().numpy() * target_scale
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    preprocessing = dict(raw_mean=raw_mean, raw_scale=raw_scale, relation_mean=rel_mean,
                         relation_scale=rel_scale, target_scale=np.asarray(target_scale))
    return predictions, state, preprocessing


def paired(names, left, right, family=1):
    videos = sorted(set(names))
    delta = np.array([(left[names == n] - right[names == n]).mean() for n in videos])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = delta[rng.integers(0, len(delta), (BOOTSTRAPS, len(delta)))].mean(1)
    low, high = np.quantile(draws, [.025, .975])
    family_low = float(np.quantile(draws, .025 / family))
    return dict(videos=len(delta), mean_delta=float(delta.mean()), ci_low=float(low), ci_high=float(high),
                family_ci_low=family_low, wins=int((delta > 1e-12).sum()),
                ties=int((np.abs(delta) <= 1e-12).sum()), losses=int((delta < -1e-12).sum()))


def summarize(data, predictions, fixed):
    q = data["q"].mean(1)
    index = np.arange(len(q))
    selected = {name: q[index, score.argmax(1)] for name, score in predictions.items()}
    selected["train_fold_fixed"] = q[index, fixed]
    selected["random_layer_expectation"] = q.mean(1)
    for action in range(6):
        selected["l" + str(action + 7)] = q[:, action]
    rows = []
    for control in selected:
        if control != "relational":
            rows.append(dict(left="relational", right=control,
                             **paired(data["names"], selected["relational"], selected[control],
                                      family=6 if control.startswith("l") else 1)))
    post = data["post_q"].mean(1)
    post_selected = {name: post[index, score.argmax(1)] for name, score in predictions.items()}
    for control in ("raw",) + tuple("l" + str(a) for a in range(7, 13)):
        value = post_selected[control] if control == "raw" else post[:, int(control[1:]) - 7]
        rows.append(dict(left="relational_h2_h20", right=control + "_h2_h20",
                         **paired(data["names"], post_selected["relational"], value, family=6 if control != "raw" else 1)))
    # Repeat draws vary only later random layers; they are NOT independent time windows.
    repeats = data["q"]
    transfer = .5 * (repeats[index, 1, repeats[:, 0].argmax(1)] +
                     repeats[index, 0, repeats[:, 1].argmax(1)])
    rows.append(dict(left="cross_repeat_hindsight", right="train_fold_fixed",
                     **paired(data["names"], transfer, selected["train_fold_fixed"])))
    lookup = {(r["left"], r["right"]): r for r in rows}
    mechanism = lookup[("cross_repeat_hindsight", "train_fold_fixed")]["ci_low"] > 0
    relation_pass = lookup[("relational", "raw")]["ci_low"] > 0
    static_pass = all(lookup[("relational", "l" + str(a))]["family_ci_low"] > 0 for a in range(7, 13))
    post_pass = (lookup[("relational_h2_h20", "raw_h2_h20")]["ci_low"] > 0 and
                 all(lookup[("relational_h2_h20", "l" + str(a) + "_h2_h20")]["family_ci_low"] > 0 for a in range(7, 13)))
    status = ("FAIL_EQUAL_COST_HEADROOM_SCREEN" if not mechanism else
              "PASS_RELATIONAL_STATE_SCREEN" if relation_pass and static_pass and post_pass else "FAIL_RELATIONAL_STATE_SCREEN")
    return rows, selected, dict(status=status, cross_repeat_headroom_pass=mechanism,
                                 relational_vs_raw_pass=relation_pass, all_six_static_family_pass=static_pass,
                                 post_anchor_transfer_pass=post_pass)


def run(directory, output, audit_only=False, device="cpu"):
    directory, output = Path(directory).expanduser().resolve(), Path(output).expanduser().resolve()
    if directory == output or directory.is_relative_to(output) or output.is_relative_to(directory):
        raise ValueError("Analysis must use a separate directory, not an ancestor of the cache")
    contract, records, missing, hashes = load_data(directory)
    print("Audited {} of {} fit clips ({} missing); tune/validation cache files not opened.".format(
        len(records), len(contract["fit"]), len(missing)), flush=True)
    if audit_only:
        result = sealed(dict(schema=SCHEMA, status="PILOT_INTEGRITY_ONLY" if missing else "PASS_BRANCH_DATA_INTEGRITY",
            contract_fingerprint=contract["fingerprint"], clips=len(records), missing=len(missing),
            maximum_replay_error=max((float(d["replay_error"].max()) for _, d in records), default=None),
            observed_anchors=ANCHORS * len(records), expected_anchors=ANCHORS * len(contract["fit"]),
            exact_cost_pass=True, relation_reconstruction_pass=True, models_trained=False,
            warning="Integrity only, not policy quality. Complete exact cache before the frozen OOF screen."))
        if not records:
            raise ValueError("No cached clips to audit")
        if any(file_hash(directory / n) != digest for n, digest in hashes.items()):
            raise RuntimeError("Collection changed during integrity audit")
        verify_code(contract)
        with single_writer(output):
            write_json(output / "stage25_data_audit.json", result)
        print("Decision: " + result["status"])
        print("H20 replay max box error: {}; exact costs and relation reconstruction passed.".format(result["maximum_replay_error"]))
        return result
    if missing:
        raise ValueError("Complete every frozen fit clip before efficacy screening; use --audit_only for smoke")
    data = dataset(records)
    if set(data["names"]) != set(contract["fit"]) or len(data["names"]) < .8 * ANCHORS * len(records):
        raise ValueError("Need eligible anchors in every fit video and >=80% of planned anchors overall")
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    screen = sealed(dict(schema=SCHEMA, role="development_only_fixed_state_probe_not_PPO",
        collection_fingerprint=contract["fingerprint"], cache_sha256=hashes, folds=FOLDS, fold_seed=FOLD_SEED,
        seeds=list(MODEL_SEEDS), epochs=EPOCHS, batch=BATCH, learning_rate=.0003, weight_decay=.001,
        hidden=128, target="six per-state centered visibility-normalized discounted IoU values",
        primary="relational", controls=["raw", "train_fold_fixed"] + ["l" + str(a) for a in range(7, 13)],
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, device=device),
        bootstrap_samples=BOOTSTRAPS, bootstrap_seed=BOOTSTRAP_SEED,
        decision="cross-repeat hindsight lower95 >0; relational-raw lower95 >0 and all six static Bonferroni lower95 >0, both H1-H20 and H2-H20",
        collection_code_sha256=contract["code_sha256"]))
    with single_writer(output):
        path = output / "stage25_screen_contract.json"
        if path.exists():
            if read_sealed(path) != screen:
                raise ValueError("Screen contract changed; do not reuse this output directory")
        elif any(p.name != ".stage23_writer.lock" for p in output.iterdir()):
            raise ValueError("Unidentified screen outputs; use a separate new directory")
        else:
            write_json(path, screen)
        predictions = {v: np.zeros((len(data["names"]), 6)) for v in ("raw", "relational")}
        fixed = np.full(len(data["names"]), -1, np.int64)
        fold_rows = []
        for fold, (train, test, test_names) in enumerate(grouped_folds(data["names"], FOLDS, FOLD_SEED)):
            weights = video_weights(data["names"][train])
            fixed[test] = int(np.average(data["q"][train].mean(1), axis=0, weights=weights).argmax())
            for variant in predictions:
                for seed in MODEL_SEEDS:
                    root = output / "folds" / str(fold) / variant / str(seed)
                    identity = dict(screen_fingerprint=screen["fingerprint"], fold=fold, variant=variant, seed=seed,
                                    train_videos=sorted(set(data["names"][train])), test_videos=list(test_names))
                    complete = root / "complete.json"
                    if complete.exists():
                        saved = read_sealed(complete)
                        if (saved["identity"] != identity or file_hash(root / "model.pt") != saved["model_sha256"]
                                or file_hash(root / "prediction.npz") != saved["prediction_sha256"]):
                            raise ValueError("Cached probe fold/checkpoint identity changed")
                        with np.load(root / "prediction.npz", allow_pickle=False) as archive:
                            score = archive["score"].copy()
                    else:
                        score, model, preprocessing = train_predict(torch, data, train, test, variant, seed,
                                                                   contract["embed_dim"], device)
                        root.mkdir(parents=True, exist_ok=True)
                        atomic_torch_save(torch, dict(identity=identity, model=model, preprocessing=preprocessing), root / "model.pt")
                        write_cache(root / "prediction.npz", dict(score=score), identity)
                        write_json(complete, sealed(dict(identity=identity, model_sha256=file_hash(root / "model.pt"),
                                                         prediction_sha256=file_hash(root / "prediction.npz"))))
                    if score.shape != (len(test), 6) or not np.isfinite(score).all():
                        raise ValueError("Invalid cached probe prediction")
                    predictions[variant][test] += score / len(MODEL_SEEDS)
                    print("[OOF {}/{}] {} seed{} complete; test={} videos".format(
                        fold + 1, FOLDS, variant, seed, len(test_names)), flush=True)
            fold_rows.append(dict(fold=fold, train_videos="|".join(sorted(set(data["names"][train]))),
                                  test_videos="|".join(test_names), training_selected_fixed_layer=int(fixed[test[0]] + 7)))
        rows, selected, decision = summarize(data, predictions, fixed)
        if any(file_hash(directory / name) != digest for name, digest in hashes.items()):
            raise RuntimeError("Collection changed during screen; no final efficacy reports published")
        verify_code(contract)
        write_csv(output / "stage25_paired.csv", rows)
        write_csv(output / "stage25_folds.csv", fold_rows)
        action_rows = [dict(sequence=str(n), anchor=int(a), raw_layer=int(predictions["raw"][i].argmax() + 7),
            relational_layer=int(predictions["relational"][i].argmax() + 7), fold_fixed_layer=int(fixed[i] + 7),
            relational_value=float(selected["relational"][i]), raw_value=float(selected["raw"][i]))
            for i, (n, a) in enumerate(zip(data["names"], data["anchor"]))]
        write_csv(output / "stage25_oof_actions.csv", action_rows)
        result = sealed(dict(schema=SCHEMA, screen_fingerprint=screen["fingerprint"], **decision,
            fit_videos=len(records), eligible_anchors=len(data["names"]), tune_labels_loaded=False,
            validation_labels_loaded=False, ppo_trained=False,
            next_step="One new matched PPO smoke may be designed; do not claim closed-loop gain" if
                decision["status"] == "PASS_RELATIONAL_STATE_SCREEN" else
                "Stop this relational-state design; do not tune on excluded tune/validation data",
            warning="Frozen representation already saw fit clips. Probe-only grouped OOF is development evidence. "
                    "Matched first-layer branches under random continuations are not PPO returns or closed-loop policy AUC. "
                    "Cross-repeat hindsight uses GT and is not an independent temporal-window test."))
        write_json(output / "stage25_decision.json", result)
        for row in rows:
            print("{left} - {right}: {mean_delta:.6f} [{ci_low:.6f},{ci_high:.6f}]".format(**row))
        print("Decision: " + result["status"])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.input.strip() or not args.output_dir.strip():
        parser.error("Explicit nonempty input/output paths required")
    run(args.input, args.output_dir, args.audit_only, args.device)


if __name__ == "__main__":
    main()
