"""Read-only GPU telemetry and an explicitly authorized, bounded load worker.

Torch is imported only inside the spawned worker. No clocks, power limits,
priorities, MPS configuration or other device-wide settings are changed.
"""

import csv
import io
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import threading
import time

from stage22_resource_common import CsvJournal, LOAD_FIELDS, load_phase


METRICS = ("utilization.gpu", "utilization.memory", "memory.used", "memory.free",
           "clocks.current.sm", "clocks.current.memory", "power.draw", "temperature.gpu")
TELEMETRY_FIELDS = ("sample_id", "start_ns", "end_ns", "gpu_uuid") + METRICS + ("error",)


def smi_query(kind, fields, gpu_uuid=None):
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required for unambiguous physical GPU identification")
    command = [executable, "--query-" + kind + "=" + ",".join(fields),
               "--format=csv,noheader,nounits"]
    if gpu_uuid:
        command += ["--id=" + gpu_uuid]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(command, capture_output=True, text=True, timeout=5,
                            check=False, creationflags=flags)
    if result.returncode:
        raise RuntimeError("nvidia-smi: " + (result.stderr or result.stdout).strip()[:500])
    rows = [[field.strip() for field in row] for row in csv.reader(io.StringIO(result.stdout)) if row]
    if any(len(row) != len(fields) for row in rows):
        raise RuntimeError("Malformed nvidia-smi query result")
    return [dict(zip(fields, row)) for row in rows]


def resolve_gpu(torch, gpu):
    """Do not confuse CUDA-visible logical indices with nvidia-smi indices."""
    props = torch.cuda.get_device_properties(gpu)
    prop_uuid = str(getattr(props, "uuid", ""))
    processes, process_error = [], ""
    try:
        processes = smi_query("compute-apps", ("pid", "gpu_uuid"))
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        process_error = str(error)
    own = {r["gpu_uuid"] for r in processes if r["pid"] == str(os.getpid())}
    if prop_uuid.startswith("GPU-"):
        gpu_uuid, mapping = prop_uuid, "torch_device_properties_uuid"
        if own and gpu_uuid not in own:
            raise RuntimeError("Torch UUID disagrees with the current process's physical GPU")
    elif len(own) == 1:
        gpu_uuid, mapping = own.pop(), "single_current_process_compute_context"
    else:
        raise RuntimeError("Cannot safely map logical CUDA device to a physical GPU UUID. "
                           "Run without MIG/MPS and with readable nvidia-smi process information.")
    fields = ("uuid", "name", "driver_version", "memory.total", "mig.mode.current")
    devices = smi_query("gpu", fields, gpu_uuid)
    if len(devices) != 1 or devices[0]["uuid"] != gpu_uuid:
        raise RuntimeError("GPU UUID query is ambiguous (MIG is not supported in this pilot)")
    if devices[0]["mig.mode.current"].lower() == "enabled" or not gpu_uuid.startswith("GPU-"):
        raise RuntimeError("Use an exclusive full GPU, not a MIG instance")
    foreign = sorted({r["pid"] for r in processes
                      if r["gpu_uuid"] == gpu_uuid and r["pid"] != str(os.getpid())})
    return dict(devices[0], logical_cuda_device=gpu, mapping=mapping,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                other_compute_pids=foreign, process_query_error=process_error,
                memory_bytes=int(props.total_memory))


def metric_number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return ""
    return number if math.isfinite(number) else ""


class TelemetrySampler:
    """Timestamp completion of every read; only completed samples are observable."""
    def __init__(self, gpu_uuid, path, interval_s=.2):
        self.gpu_uuid, self.interval_s = gpu_uuid, interval_s
        self.journal = CsvJournal(path, TELEMETRY_FIELDS)
        self.rows, self.errors = [], []
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.thread = None

    def sample(self):
        start = time.monotonic_ns()
        values, error = {}, ""
        try:
            rows = smi_query("gpu", METRICS, self.gpu_uuid)
            if len(rows) != 1:
                raise RuntimeError("Telemetry query did not return one GPU")
            values = {key: metric_number(value) for key, value in rows[0].items()}
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            error = str(exc)
        end = time.monotonic_ns()
        with self.lock:
            row = dict(sample_id=len(self.rows), start_ns=start, end_ns=end, gpu_uuid=self.gpu_uuid)
            row.update({k: values.get(k, "") for k in METRICS})
            row["error"] = error
            self.rows.append(row)
            self.journal.append(row)
            self.journal.flush()
            if error:
                self.errors.append(error)

    def start(self):
        self.sample()
        def loop():
            while not self.stop_event.wait(self.interval_s):
                self.sample()
        self.thread = threading.Thread(target=loop, name="stage22-telemetry", daemon=True)
        self.thread.start()

    def latest(self, before_ns):
        with self.lock:
            for row in reversed(self.rows):
                if row["end_ns"] <= before_ns and not row["error"]:
                    return row.copy()
        return None

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=7)
            if self.thread.is_alive():
                raise RuntimeError("Telemetry sampler did not stop")
        self.journal.close()


def load_worker(connection, stop_event, config, condition, gpu, gpu_uuid, path):
    """One synchronized matmul at a time; never submit an unbounded CUDA queue."""
    journal = None
    try:
        parent = mp.parent_process()
        if parent is not None and not parent.is_alive():
            return
        import torch
        torch.set_num_threads(1)
        torch.cuda.set_device(gpu)
        torch.manual_seed(config["seed"])
        device = torch.device("cuda", gpu)
        size = config["matrix_size"]
        x = torch.randn((size, size), device=device)
        y = torch.randn_like(x)
        z = torch.empty_like(x)
        identity = resolve_gpu(torch, gpu)
        if identity["uuid"] != gpu_uuid:
            raise RuntimeError("Contention worker landed on a different physical GPU")
        def kernel():
            start = time.monotonic_ns()
            torch.mm(x, y, out=z)
            torch.cuda.synchronize(gpu)
            return (time.monotonic_ns() - start) / 1e6
        with torch.no_grad():
            for _ in range(5):
                kernel()
            idle_ms = max(kernel() for _ in range(10))
        if idle_ms > config["maximum_idle_kernel_ms"]:
            raise RuntimeError("Load kernel exceeds the frozen idle safety bound: %.3f ms. "
                               "Use a new run with a smaller --matrix_size." % idle_ms)
        connection.send(dict(status="ready", gpu_uuid=gpu_uuid, pid=os.getpid(),
                             idle_max_kernel_ms=idle_ms, matrix_size=size,
                             torch=str(torch.__version__), cuda=str(torch.version.cuda),
                             matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                             allocated_bytes=int(torch.cuda.memory_allocated(gpu))))
        origin_ns = connection.recv()
        journal = CsvJournal(path, LOAD_FIELDS)
        period_s = config["load_period_ms"] / 1000
        with torch.no_grad():
            while not stop_event.is_set() and (parent is None or parent.is_alive()):
                start = time.monotonic_ns()
                elapsed = max(0., (start - origin_ns) / 1e9)
                phase, duty = load_phase(condition, elapsed, config)
                budget_ns = int(period_s * duty * 1e9)
                kernels, maximum = 0, 0.
                busy_start = time.monotonic_ns()
                while duty > 0 and time.monotonic_ns() - busy_start < budget_ns:
                    if stop_event.is_set():
                        break
                    maximum = max(maximum, kernel())
                    kernels += 1
                busy_ms = (time.monotonic_ns() - busy_start) / 1e6 if kernels else 0.
                remainder = period_s - (time.monotonic_ns() - start) / 1e9
                if remainder > 0:
                    stop_event.wait(remainder)
                journal.append(dict(start_ns=start, end_ns=time.monotonic_ns(),
                                    elapsed_s=elapsed, phase=phase, nominal_duty=duty,
                                    kernels=kernels, busy_ms=busy_ms, maximum_kernel_ms=maximum))
                journal.flush()
        connection.send(dict(status="stopped", cycles=journal.count))
    except BaseException as error:
        try:
            connection.send(dict(status="error", error=repr(error)))
        except (OSError, BrokenPipeError):
            pass
    finally:
        if journal is not None:
            journal.close()
        connection.close()


class LoadProcess:
    def __init__(self, config, condition, gpu, gpu_uuid, path):
        ctx = mp.get_context("spawn")
        self.parent, child = ctx.Pipe()
        self.stop_event = ctx.Event()
        self.process = ctx.Process(target=load_worker, args=(child, self.stop_event, config,
                                   condition, gpu, gpu_uuid, str(path)), daemon=True)
        self.child = child
        self.ready = None
        self.stopped = False

    def start(self):
        self.process.start()
        self.child.close()
        if not self.parent.poll(90):
            raise RuntimeError("Contention worker startup timed out")
        self.ready = self.parent.recv()
        if self.ready.get("status") != "ready":
            raise RuntimeError("Contention worker failed: " + str(self.ready))
        origin = time.monotonic_ns()
        self.parent.send(origin)
        return origin

    def check(self):
        if not self.process.is_alive() or self.parent.poll():
            message = self.parent.recv() if self.parent.poll() else {"exitcode": self.process.exitcode}
            raise RuntimeError("Contention worker stopped during profiling: " + str(message))

    def close(self):
        if self.stopped:
            return
        self.stopped = True
        self.stop_event.set()
        message = None
        try:
            if self.process.pid is not None:
                if self.parent.poll(15):
                    try:
                        message = self.parent.recv()
                    except EOFError:
                        message = {"status": "error", "error": "worker pipe closed"}
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
                    raise RuntimeError("Contention worker required forced termination; run is invalid")
                if message is None or message.get("status") != "stopped":
                    raise RuntimeError("Contention worker did not exit cleanly: " + str(message))
        finally:
            self.parent.close()
            self.child.close()
