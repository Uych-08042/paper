"""Launch sealed Stage-61 with its recorded config and explicit identity diagnostics.

Only parameter construction is adapted, in memory. The original runtime factory,
checkpoint hash, complete runtime identity and original model-state hash must match.
No sealed experiment file or manifest is edited.
"""
from __future__ import annotations
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tracking") not in sys.path:
    sys.path.insert(0, str(REPO / "tracking"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from stage4_common import file_hash, fingerprint, write_json


def gpu_inventory(plan, torch):
    """CUDA logical indices, including visibility remapping; no model or images."""
    expected=plan["runtime_identity"]
    devices=[]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append(dict(index=index,name=torch.cuda.get_device_name(index),
                uuid=str(getattr(torch.cuda.get_device_properties(index),"uuid","unavailable"))))
    exact=[r["index"] for r in devices if r["name"]==expected["tracker"]["device"] and r["uuid"]==expected["gpu_uuid"]]
    return dict(source_device=expected["tracker"]["device"],source_gpu_uuid=expected["gpu_uuid"],
        visible_cuda_devices=devices,exact_source_indices=exact,
        same_model_indices=[r["index"] for r in devices if r["name"]==expected["tracker"]["device"]])


def source_gpu(inventory):
    matches=inventory["exact_source_indices"]
    if len(matches)!=1:
        raise ValueError("Original source GPU is not uniquely visible. Auto selection never substitutes a different GPU.\n"+
                         json.dumps(inventory,ensure_ascii=False,indent=2))
    return matches[0]


def differences(expected, actual, prefix="runtime"):
    """Value comparisons retain the original dictionary-equality semantics."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result=[]
        for key in sorted(set(expected) | set(actual)):
            path=prefix+"."+str(key)
            if key not in expected: result.append(dict(field=path,expected="<absent>",actual=actual[key]))
            elif key not in actual: result.append(dict(field=path,expected=expected[key],actual="<absent>"))
            else: result.extend(differences(expected[key],actual[key],path))
        return result
    if isinstance(expected,(list,tuple)) and isinstance(actual,(list,tuple)) and type(expected)==type(actual):
        if len(expected)!=len(actual): return [dict(field=prefix,expected=expected,actual=actual)]
        return [r for i,(left,right) in enumerate(zip(expected,actual)) for r in differences(left,right,prefix+"[{}]".format(i))]
    return [] if expected==actual else [dict(field=prefix,expected=expected,actual=actual)]


def frozen_parameters(expected, checkpoint, parameter, params_type=None):
    if parameter!=expected["parameter"]: raise ValueError("Parameter name differs from sealed source")
    from easydict import EasyDict
    if params_type is None:
        from lib.test.utils import TrackerParams
        params_type=TrackerParams
    params=params_type()
    params.cfg=EasyDict(copy.deepcopy(expected["config"]))
    params.template_factor=params.cfg.TEST.TEMPLATE_FACTOR
    params.template_size=params.cfg.TEST.TEMPLATE_SIZE
    params.search_factor=params.cfg.TEST.SEARCH_FACTOR
    params.search_size=params.cfg.TEST.SEARCH_SIZE
    params.checkpoint=str(Path(checkpoint).expanduser().resolve())
    params.save_all_boxes=False
    return params


def checked_runtime(args, plan, root, maker=None, parameter_module=None, runtime_module=None, params_type=None):
    """Report actual differences; never suppress a weights/software/device mismatch."""
    import torch
    from stage46_motion_experiment import make_runtime
    maker=maker or make_runtime
    expected=plan["runtime"]
    diagnostic=dict(schema="stage61_runtime_replay_diagnostic_v1",role="startup_identity_only",
        plan_fingerprint=plan["fingerprint"],launcher_sha256=file_hash(__file__),mode=args.mode,
        config_source="sealed manifest runtime.config",current_yaml_differences=[],mismatches=[],status="CHECKING")
    path=Path(root)/"diagnostics"/("stage61_runtime_"+args.mode+".json")

    def publish(status):
        diagnostic["status"]=status
        write_json(path,diagnostic)

    def reject(rows):
        diagnostic["mismatches"]=rows; publish("RUNTIME_MISMATCH")
        detail="\n".join("  {}: expected={} actual={}".format(r["field"],repr(r["expected"]),repr(r["actual"])) for r in rows[:30])
        if "gpu_inventory" in diagnostic:
            detail+="\nVisible CUDA GPUs: "+json.dumps(diagnostic["gpu_inventory"],ensure_ascii=False)
        raise ValueError("Stage-61 runtime identity mismatch; no image inference started:\n"+detail+"\nDiagnostic: "+str(path))

    bound=differences(plan["runtime_identity"]["tracker"],expected,"sealed.runtime")
    if bound: reject(bound)
    checkpoint=Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file(): raise FileNotFoundError("Original checkpoint missing: "+str(checkpoint))
    # Reject the wrong checkpoint before attempting to build a possibly incompatible model.
    actual=dict(parameter=args.parameter,checkpoint_sha256=file_hash(checkpoint),torch=str(torch.__version__),cuda=str(torch.version.cuda))
    subset={key:expected[key] for key in actual}
    bad=differences(subset,actual,"tracker")
    if bad: reject(bad)
    if not torch.cuda.is_available(): reject([dict(field="tracker.cuda_available",expected=True,actual=False)])
    diagnostic["gpu_inventory"]=gpu_inventory(plan,torch)
    indices=[r["index"] for r in diagnostic["gpu_inventory"]["visible_cuda_devices"]]
    if args.gpu not in indices: reject([dict(field="tracker.cuda_logical_index",expected=indices,actual=args.gpu)])
    device=torch.cuda.get_device_name(args.gpu)
    if device!=expected["device"]: reject([dict(field="tracker.device",expected=expected["device"],actual=device)])
    parameter_module=parameter_module or importlib.import_module("lib.test.parameter.sglatrack")
    runtime_module=runtime_module or importlib.import_module("stage4_runtime")
    # Diagnose mutable YAML/global config drift; use an independent sealed config afterwards.
    try:
        current=parameter_module.parameters(args.parameter)
        diagnostic["current_yaml_differences"]=differences(expected["config"],copy.deepcopy(current.cfg),"config")
    except Exception as error:
        diagnostic["current_yaml_probe_error"]=type(error).__name__+": "+str(error)
    diagnostic["frozen_config_restored"]=bool(diagnostic["current_yaml_differences"] or diagnostic.get("current_yaml_probe_error"))
    if diagnostic["frozen_config_restored"]:
        print("[Stage-61 startup] Using the sealed source config; current YAML/global config differs or cannot be read.",flush=True)
        for row in diagnostic["current_yaml_differences"][:20]:
            print("  {}: source={} current={}".format(row["field"],row["expected"],row["actual"]),flush=True)
    original_runtime=runtime_module.Runtime

    def build(*values,**keywords):
        rt=original_runtime(*values,**keywords)
        bad=differences(expected,rt.identity,"tracker")
        if bad: reject(bad)
        return rt

    def parameters(name):
        return frozen_parameters(expected,checkpoint,name,params_type)

    publish("CHECKING")
    try:
        with patch.object(parameter_module,"parameters",parameters),patch.object(runtime_module,"Runtime",build):
            native,identity=maker(args,dict(runtime=expected))
    except Exception as error:
        if diagnostic["status"]=="CHECKING":
            diagnostic["error"]=type(error).__name__+": "+str(error)
            publish("RUNTIME_CONSTRUCTION_FAILED")
        raise
    bad=differences(plan["runtime_identity"],identity)
    if bad: reject(bad)
    diagnostic["verified_runtime_fingerprint"]=fingerprint(identity)
    publish("EXACT_SOURCE_RUNTIME_VERIFIED")
    print("[Stage-61 startup] Exact checkpoint/config/software/GPU/model-state identity verified.",flush=True)
    return native,identity


def main():
    import stage61_search_common as common
    import stage61_search_experiment as experiment
    import stage46_motion_experiment as legacy
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode"); parser.add_argument("--input")
    parser.add_argument("--gpu")
    parser.add_argument("--list-gpus",action="store_true")
    known,_=parser.parse_known_args()
    if known.list_gpus:
        if not known.input: raise ValueError("--list-gpus requires --input Stage-61 directory")
        import torch
        plan=common.read_sealed(Path(known.input).expanduser().resolve()/"manifest.json")
        print(json.dumps(gpu_inventory(plan,torch),ensure_ascii=False,indent=2)); return
    if known.mode not in ("smoke","evaluate"):
        experiment.main(); return
    if not known.input: raise ValueError("Need --input Stage-61 output directory")
    root=Path(known.input).expanduser().resolve()
    # Full frozen source/code checks still run inside experiment.main before this adapter is called.
    plan=common.read_sealed(root/"manifest.json")
    if known.gpu=="auto":
        import torch
        index=source_gpu(gpu_inventory(plan,torch))
        values=list(sys.argv)
        for i,value in enumerate(values):
            if value=="--gpu": values[i+1]=str(index)
            elif value=="--gpu=auto": values[i]="--gpu="+str(index)
        print("[Stage-61 startup] Original source GPU found at CUDA logical index "+str(index),flush=True)
    else: values=list(sys.argv)
    maker=legacy.make_runtime

    def launch(args,source):
        if source["runtime"]!=plan["runtime"]: raise ValueError("Caller/source runtime binding changed")
        return checked_runtime(args,plan,root,maker=maker)

    with patch.object(legacy,"make_runtime",launch),patch.object(sys,"argv",values): experiment.main()


if __name__=="__main__": main()
