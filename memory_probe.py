"""Per-prompt peak memory, reported to Cubric Studio (MPI-1448).

A daemon thread watches the prompt queue. While a prompt runs it samples this process's RAM,
the system's RAM and the GPU's memory every SAMPLE_S; when the prompt leaves the queue it sends
one `mpi.memory_peak` websocket message to the client that queued it. The app turns those peaks
into the longest clip each model can make on this machine.

No node: it runs for every prompt, so a benchmark and an ordinary job are measured the same way.
The Pod wrapper relays every text frame, so the same message reaches the app from a Pod.
"""
import threading
import time

import psutil

EVENT = "mpi.memory_peak"
SAMPLE_S = 0.25   # a VAE decode spike lasts seconds; a quarter second catches it
IDLE_S = 0.5      # how often an idle engine checks for a new prompt

_started = False


def _gpu():
    """(device, total bytes, name) of the GPU ComfyUI runs on, or (None, 0, None) on a CPU engine."""
    import comfy.model_management as mm  # type: ignore
    dev = mm.get_torch_device()
    if getattr(dev, "type", "cpu") == "cpu":
        return None, 0, None
    name = None
    if dev.type == "cuda":
        import torch  # type: ignore
        name = torch.cuda.get_device_name(dev)
    return dev, int(mm.get_total_memory(dev)), name


def _vram_used(dev, total):
    """Bytes of GPU memory taken: ComfyUI's own free figure counts torch's reserved-but-idle cache as free."""
    import comfy.model_management as mm  # type: ignore
    return total - int(mm.get_free_memory(dev))


def _proc_ram(proc):
    """This process's memory: private bytes on Windows (what the page file must back), RSS elsewhere."""
    mi = proc.memory_info()
    return int(getattr(mi, "private", mi.rss))


def _sys_ram_used():
    vm = psutil.virtual_memory()
    return int(vm.total - vm.available)


class Probe:
    """One running prompt's peaks. Pure bookkeeping, so tests drive it without an engine."""

    def __init__(self, prompt_id, client_id, vram_used, sys_ram, proc_ram):
        self.prompt_id = prompt_id
        self.client_id = client_id
        self.t0 = time.monotonic()
        self.start = {"vram": vram_used, "sys_ram": sys_ram, "proc_ram": proc_ram}
        self.peak = dict(self.start)
        self.samples = 0

    def sample(self, vram_used, sys_ram, proc_ram):
        self.samples += 1
        for key, val in (("vram", vram_used), ("sys_ram", sys_ram), ("proc_ram", proc_ram)):
            if val is not None and (self.peak[key] is None or val > self.peak[key]):
                self.peak[key] = val

    def report(self, history_entry, totals, gpu_name):
        status = (history_entry or {}).get("status") or {}
        exc = None
        for kind, data in status.get("messages") or []:
            if kind == "execution_error":
                exc = (data or {}).get("exception_type")
        return {
            "prompt_id": self.prompt_id,
            "status": status.get("status_str"),      # "success" | "error" | None (interrupted / unknown)
            "exception_type": exc,
            "seconds": round(time.monotonic() - self.t0, 2),
            "samples": self.samples,
            "start": self.start,
            "peak": self.peak,
            "totals": totals,
            "gpu": gpu_name,
        }


def _running(queue):
    """(prompt_id, client_id) of the prompt executing now, or None."""
    with queue.mutex:
        items = list(queue.currently_running.values())
    if not items:
        return None
    item = items[0]
    return item[1], (item[3] or {}).get("client_id")


def _loop(server):
    proc = psutil.Process()
    dev, vram_total, gpu_name = _gpu()
    totals = {"ram": int(psutil.virtual_memory().total), "vram": vram_total}

    def read():
        vram = _vram_used(dev, vram_total) if dev is not None else None
        return vram, _sys_ram_used(), _proc_ram(proc)

    cur = None
    while True:
        queue = getattr(server, "prompt_queue", None)
        now = _running(queue) if queue is not None else None
        if cur is not None and (now is None or now[0] != cur.prompt_id):
            entry = queue.get_history(prompt_id=cur.prompt_id).get(cur.prompt_id)
            server.send_sync(EVENT, cur.report(entry, totals, gpu_name), cur.client_id)
            cur = None
        if cur is None and now is not None:
            cur = Probe(now[0], now[1], *read())
        if cur is not None:
            cur.sample(*read())
        time.sleep(SAMPLE_S if cur is not None else IDLE_S)


def start():
    """Start the sampler once per engine process."""
    global _started
    if _started:
        return
    _started = True
    from server import PromptServer  # type: ignore
    threading.Thread(target=_loop, args=(PromptServer.instance,), name="mpi-memory-probe", daemon=True).start()
