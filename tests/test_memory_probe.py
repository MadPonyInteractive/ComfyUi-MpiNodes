"""CPU checks for memory_probe (Cubric Studio MPI-1448): one `mpi.memory_peak` per prompt, sent to the
client that queued it, carrying the highest reading of each kind and the prompt's outcome.

No engine needed (any python with psutil):

    python tests/test_memory_probe.py
"""
import importlib
import os
import sys
import threading
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg = types.ModuleType("mpinodes")
pkg.__path__ = [ROOT]
sys.modules["mpinodes"] = pkg
mp = importlib.import_module("mpinodes.memory_probe")


class _Stop(Exception):
    pass


class FakeQueue:
    def __init__(self, timeline, history):
        self.mutex = threading.RLock()
        self.timeline = timeline   # one entry per loop pass: None or (prompt_id, client_id)
        self.history = history
        self.tick = 0

    @property
    def currently_running(self):
        now = self.timeline[min(self.tick, len(self.timeline) - 1)]
        return {} if now is None else {0: (1, now[0], {}, {"client_id": now[1]}, [])}

    def get_history(self, prompt_id=None):
        return {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}


class FakeServer:
    def __init__(self, queue):
        self.prompt_queue = queue
        self.sent = []

    def send_sync(self, event, data, sid=None):
        self.sent.append((event, data, sid))


def run(timeline, readings, history):
    """Drive _loop one pass per timeline entry; readings[i] = (vram, sys_ram, proc_ram) for read i."""
    queue = FakeQueue(timeline, history)
    server = FakeServer(queue)
    reads = iter(readings)
    last = {}

    def vram(dev, total):   # read() asks vram first, then system RAM, then process RAM
        last["r"] = next(reads)
        return last["r"][0]
    mp._gpu = lambda: ("dev", 24 << 30, "Fake GPU")
    mp._vram_used = vram
    mp._sys_ram_used = lambda: last["r"][1]
    mp._proc_ram = lambda proc: last["r"][2]

    def sleep(_s):
        queue.tick += 1
        if queue.tick >= len(timeline):
            raise _Stop()
    mp.time.sleep = sleep
    try:
        mp._loop(server)
    except _Stop:
        pass
    return server.sent


def test_one_prompt_reports_its_peaks_once():
    ok = {"status": {"status_str": "success", "messages": []}}
    sent = run(
        timeline=[("p1", "c1"), ("p1", "c1"), ("p1", "c1"), None, None],
        # the start read, then one sample per running pass
        readings=[(1, 10, 100), (1, 10, 100), (5, 30, 200), (3, 50, 150)],
        history={"p1": ok},
    )
    assert len(sent) == 1, sent
    event, data, sid = sent[0]
    assert event == "mpi.memory_peak" and sid == "c1"
    assert data["prompt_id"] == "p1" and data["status"] == "success"
    assert data["start"] == {"vram": 1, "sys_ram": 10, "proc_ram": 100}
    assert data["peak"] == {"vram": 5, "sys_ram": 50, "proc_ram": 200}
    assert data["totals"]["vram"] == 24 << 30 and data["gpu"] == "Fake GPU"


def test_back_to_back_prompts_and_a_failure():
    err = {"status": {"status_str": "error", "messages": [
        ["execution_start", {}], ["execution_error", {"exception_type": "torch.OutOfMemoryError"}]]}}
    ok = {"status": {"status_str": "success", "messages": []}}
    sent = run(
        timeline=[("a", "c1"), ("b", "c2"), None, None],
        readings=[(1, 1, 1), (2, 2, 2), (7, 7, 7), (8, 8, 8)],
        history={"a": err, "b": ok},
    )
    assert [(d["prompt_id"], s) for _, d, s in sent] == [("a", "c1"), ("b", "c2")], sent
    assert sent[0][1]["status"] == "error" and sent[0][1]["exception_type"] == "torch.OutOfMemoryError"
    assert sent[0][1]["peak"]["vram"] == 2
    assert sent[1][1]["peak"]["vram"] == 8


def test_cpu_engine_has_no_vram():
    p = mp.Probe("x", None, None, 5, 6)
    p.sample(None, 9, 4)
    assert p.peak == {"vram": None, "sys_ram": 9, "proc_ram": 6}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
