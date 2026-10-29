"""Prepare, smoke, train, resume and evaluate the small Stage-45 detection pilot."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path

import stage45_detection_common as c


@contextmanager
def run_lock(root):
    path = Path(root) / "run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("A Stage-45 job may still be running. Inspect run.lock/PID before removing a stale lock.") from exc
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def atomic_torch(torch, path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_update(rt, plan, annotations, step, optimizer):
    _, frames = c.schedule(plan, step)
    rt.reset()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, frame in enumerate(frames):
        image = rt.image(frame)
        if index < c.DESIGN["burn_in"]:
            with rt.torch.no_grad():
                rt.forward(image)
        else:
            batch = rt.target(image, annotations[frame["id"]])
            _, loss, _ = rt.forward(image, batch)
            if not rt.torch.isfinite(loss):
                raise FloatingPointError("Non-finite gate training loss")
            (loss / (len(frames)-c.DESIGN["burn_in"])).backward()
            losses.append(float(loss.detach()))
    parameters = list(rt.gates.parameters())
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters):
        raise ValueError("No gate gradient")
    if any(p.grad is not None and not rt.torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("Non-finite gate gradient")
    norm = rt.torch.nn.utils.clip_grad_norm_(parameters, c.DESIGN["grad_clip"])
    optimizer.step()
    return dict(step=step+1, loss=sum(losses)/len(losses), gradient_norm=float(norm))


def smoke(rt, plan, root):
    from stage45_detection_runtime import tensor_digest
    torch = rt.torch
    destination = Path(root) / "smoke/decision.json"
    if destination.exists():
        old = c.read_sealed(destination)
        if old["plan"] != plan["fingerprint"] or old["runtime"] != rt.identity:
            raise ValueError("Mixed smoke plan/runtime")
        print("[reuse] Stage-45 smoke", old["status"])
        return old
    video = plan["videos"]["fit"][0]
    frames = video["frames"][:c.DESIGN["smoke_frames"]]
    rt.configure("native")
    original = []
    with torch.no_grad():
        for frame in frames:
            _, _, decoded = rt.forward(rt.image(frame))
            original.append(decoded.cpu().clone())
    errors = {}
    updates = {}
    count = []
    _, annotations = c.annotation_index(c.read_json(plan["annotations"]["fit"]["path"]))
    for arm in ("capacity", "signals"):
        rt.configure(arm)
        count.append(sum(p.numel() for p in rt.gates.parameters()))
        maximum = 0.
        with torch.no_grad():
            for frame, expected in zip(frames, original):
                _, _, decoded = rt.forward(rt.image(frame))
                maximum = max(maximum, float((decoded.cpu()-expected).abs().max()))
        if maximum != 0:
            raise ValueError("Identity adapter did not reproduce native decoded predictions exactly")
        errors[arm] = maximum
        before = tensor_digest(rt.gates.state_dict())
        optimizer = torch.optim.AdamW(rt.gates.parameters(), lr=c.DESIGN["lr"], weight_decay=0.)
        updates[arm] = train_update(rt, plan, annotations, 0, optimizer)
        if before == tensor_digest(rt.gates.state_dict()):
            raise ValueError("Smoke optimizer did not change gates")
        rt.reset()
        with torch.no_grad():
            _, _, first = rt.forward(rt.image(frames[0]))
        if not torch.equal(first.cpu(), original[0]):
            raise ValueError("Trained gate altered the native first-frame reset")
    if len(set(count)) != 1:
        raise ValueError("Gate parameter budgets differ")
    rt.assert_frozen()
    record = c.seal(dict(schema=c.SCHEMA, status="PASS_DETECTION_GATE_SMOKE", plan=plan["fingerprint"],
        runtime=rt.identity, native_decode_errors=errors, trainable_parameters=count[0], updates=updates,
        training_split="fit only", detector_unchanged=True, method_gain_established=False))
    c.atomic_json(Path(root) / "smoke" / "decision.json", record, immutable=True)
    print("Stage-45 smoke PASS: native parity, matched capacity, gate gradients and frozen detector")
    return record


def require_smoke(plan, root, runtime=None):
    record = c.read_sealed(Path(root) / "smoke/decision.json")
    if record["plan"] != plan["fingerprint"] or record["status"] != "PASS_DETECTION_GATE_SMOKE":
        raise ValueError("Need smoke for this exact plan")
    if runtime is not None and record["runtime"] != runtime:
        raise ValueError("Smoke runtime differs; keep the verified environment/device")


def checkpoint_path(root, arm, step):
    return Path(root) / "training" / arm / ("step{:04d}.pt".format(step))


def load_checkpoint(rt, path, plan, arm):
    path = Path(path)
    receipt = c.read_sealed(path.with_suffix(".receipt.json"))
    if receipt["sha256"] != c.digest(path):
        raise ValueError("Stage-45 checkpoint changed: " + str(path))
    state = rt.torch.load(path, map_location="cpu", weights_only=True)
    if state["plan"] != plan["fingerprint"] or state["arm"] != arm or state["runtime"] != rt.identity:
        raise ValueError("Mixed checkpoint plan/arm/runtime")
    if (receipt.get("plan") != plan["fingerprint"] or receipt.get("arm") != arm or
        receipt.get("step") != state["step"] or path.name != "step{:04d}.pt".format(state["step"]) or
        len(state["history"]) != state["step"]):
        raise ValueError("Mixed checkpoint receipt/step/history")
    return state


def train(rt, plan, root, arm):
    require_smoke(plan, root, rt.identity)
    rt.configure(arm)
    optimizer = rt.torch.optim.AdamW(rt.gates.parameters(), lr=c.DESIGN["lr"], weight_decay=c.DESIGN["weight_decay"])
    existing = sorted((Path(root) / "training" / arm).glob("step*.receipt.json"))
    history, start = [], 0
    if existing:
        path = Path(str(existing[-1]).replace(".receipt.json", ".pt"))
        state = load_checkpoint(rt, path, plan, arm)
        rt.gates.load_state_dict(state["gates"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        history, start = state["history"], state["step"]
    _, annotations = c.annotation_index(c.read_json(plan["annotations"]["fit"]["path"]))
    for step in range(start, c.DESIGN["updates"]):
        history.append(train_update(rt, plan, annotations, step, optimizer))
        if (step+1) % c.DESIGN["save_every"] == 0 or step+1 == c.DESIGN["updates"]:
            rt.assert_frozen()
            path = checkpoint_path(root, arm, step+1)
            payload = dict(plan=plan["fingerprint"], arm=arm, runtime=rt.identity,
                           step=step+1, gates=rt.gates.state_dict(), optimizer=optimizer.state_dict(), history=history)
            atomic_torch(rt.torch, path, payload)
            c.atomic_json(path.with_suffix(".receipt.json"), c.seal(dict(schema=c.SCHEMA,
                sha256=c.digest(path), plan=plan["fingerprint"], arm=arm, step=step+1)), immutable=True)
            print("[{}] {}/{} loss={:.6f}".format(arm, step+1, c.DESIGN["updates"], history[-1]["loss"]), flush=True)


def evaluate(rt, plan, root, sequence=None):
    require_smoke(plan, root, rt.identity)
    models = {arm: load_checkpoint(rt, checkpoint_path(root, arm, c.DESIGN["updates"]), plan, arm)
              for arm in ("capacity", "signals")}
    bindings = {arm: c.digest(checkpoint_path(root, arm, c.DESIGN["updates"])) for arm in models}
    run = c.seal(dict(schema=c.SCHEMA, plan=plan["fingerprint"], runtime=rt.identity, models=bindings))
    c.atomic_json(Path(root) / "evaluation/run.json", run, immutable=True)
    videos = plan["videos"]["checking"]
    if sequence is not None:
        videos = [v for v in videos if v["video_id"] == str(sequence)]
        if not videos:
            raise ValueError("Unknown checking video")
    for arm in c.ARMS:
        rt.configure(arm, models[arm]["gates"] if arm in models else None)
        for video in videos:
            path = c.result_path(root, arm, video)
            meta = dict(run=run["fingerprint"], arm=arm, video_id=video["video_id"])
            if c.committed(path, meta) is not None:
                print("[reuse]", arm, video["video_id"])
                continue
            rt.reset()
            records = []
            with rt.torch.no_grad():
                for i, frame in enumerate(video["frames"]):
                    record, _, _ = rt.forward(rt.image(frame))
                    records.append(dict(record, image_id=frame["id"], frame_id=frame["frame_id"]))
                    if (i+1) % 250 == 0:
                        print("[{}] video={} {}/{}".format(arm, video["video_id"], i+1, len(video["frames"])), flush=True)
            rt.assert_frozen()
            c.commit_result(path, dict(meta=meta, frames=records), meta)
            print("[done]", arm, video["video_id"], len(records), flush=True)


def status(plan, root):
    print("Stage-45: fixed seed={} 8-fit / 8-checking video proxies; pretrained overlap".format(c.DESIGN["seed"]))
    for arm in c.ARMS:
        print(arm, "final_model=" + str(arm == "native" or checkpoint_path(root, arm, c.DESIGN["updates"]).exists()),
              "committed_videos=" + str(sum(c.result_path(root, arm, v).with_suffix(".receipt.json").exists()
                                          for v in plan["videos"]["checking"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "smoke", "train", "evaluate", "status"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--yoloft_root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--arm", choices=("capacity", "signals"))
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.mode == "prepare":
        if any(getattr(args, k) is None for k in ("data_root", "yoloft_root", "weights", "output_dir")):
            parser.error("prepare needs --data_root --yoloft_root --weights --output_dir")
        c.prepare(args.data_root, args.yoloft_root, args.weights, args.output_dir)
        return
    if args.input is None:
        parser.error("Need --input Stage-45 root")
    root = args.input.resolve()
    plan = c.load_plan(root)
    if args.mode == "status":
        status(plan, root)
        return
    from stage45_detection_runtime import Runtime
    with run_lock(root):
        rt = Runtime(plan, root, args.gpu)
        if args.mode == "smoke":
            smoke(rt, plan, root)
        elif args.mode == "train":
            if args.arm is None:
                parser.error("train needs --arm capacity or signals")
            train(rt, plan, root, args.arm)
        else:
            evaluate(rt, plan, root, args.sequence)


if __name__ == "__main__":
    main()
