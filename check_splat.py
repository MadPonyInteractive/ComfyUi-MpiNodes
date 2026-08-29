"""Exercise MpiBrushTrain's logic with no ComfyUI, no GPU and no network.

Run it with the ComfyUI portable interpreter, which has torch — help_funcs needs it:

    <ComfyUI>/python_embeded/python.exe check_splat.py

splat.py imports folder_paths / comfy.* at module level (the pack's convention), so
those are stubbed here. What is actually asserted is the logic with no precedent in
this pack: the export-directory progress parse (Brush prints nothing off a TTY),
the single-model dataset staging, and the failure/cancel paths of the poll loop —
a bake runs for tens of minutes on a GPU, so every one of those must be right
before anyone waits on it.

Not covered, deliberately: whether Brush actually trains. That needs a real dataset
and a real GPU, and it is the bench verification, not this file.
"""
import os, sys, types, tempfile, shutil

PACK = os.path.dirname(os.path.abspath(__file__))

# Import splat.py as a SUBMODULE of a synthetic package. Two reasons: the pack's
# files use relative imports (`from .help_funcs import ...`), and putting PACK on
# sys.path directly would let the pack's own json.py shadow the stdlib module,
# which breaks help_funcs on import. The synthetic package also skips the pack's
# real __init__.py, which would drag in torch.
_pkg = types.ModuleType("mpinodes")
_pkg.__path__ = [PACK]
sys.modules["mpinodes"] = _pkg

fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: tempfile.gettempdir()
sys.modules["folder_paths"] = fp

comfy = types.ModuleType("comfy")
utils = types.ModuleType("comfy.utils")
class _PBar:
    def __init__(self, total): self.total, self.n = total, 0
    def update(self, d): self.n += d
utils.ProgressBar = _PBar
mm = types.ModuleType("comfy.model_management")
mm.throw_exception_if_processing_interrupted = lambda: None
comfy.utils, comfy.model_management = utils, mm
sys.modules.update({"comfy": comfy, "comfy.utils": utils, "comfy.model_management": mm})

import importlib
splat = importlib.import_module("mpinodes.splat")

# --- exported_step: this is what stands in for progress, so it must not guess.
assert splat.exported_step("export_15000.ply") == 15000
assert splat.exported_step("export_0.ply") == 0
assert splat.exported_step("scene.ply") is None, "a non-step .ply must not read as step 0-something"
assert splat.exported_step("export_final.ply") is None
print("ok  exported_step")

# --- the max() over exported steps is how the FINAL ply is chosen. 30000 must beat
#     5000 numerically, not lexically — "5000" > "30000" as strings.
names = ["export_5000.ply", "export_30000.ply", "export_10000.ply"]
assert max(names, key=lambda f: splat.exported_step(f) or 0) == "export_30000.ply"
assert sorted(names)[-1] != "export_30000.ply", "guard is meaningful: lexical sort picks wrong"
print("ok  final ply chosen numerically, not lexically")

# --- staging: a clean dataset must be handed back untouched, a SplatKit one staged.
root = tempfile.mkdtemp(prefix="splatcheck-")
try:
    clean_ds = os.path.join(root, "already_clean")
    os.makedirs(os.path.join(clean_ds, "sparse", "0"))
    assert splat.stage_clean_dataset(clean_ds) == clean_ds, "no _spheresfm_work = nothing to stage"
    print("ok  a single-model dataset is passed through untouched")

    ds = os.path.join(root, "splatkit")
    os.makedirs(os.path.join(ds, "_spheresfm_work", "sparse", "0"))
    os.makedirs(os.path.join(ds, "sparse", "0"))
    os.makedirs(os.path.join(ds, "images"))
    os.makedirs(os.path.join(ds, "masks"))
    for i in range(3):
        open(os.path.join(ds, "images", f"frame_{i:03d}.png"), "w").write(f"img{i}")
    open(os.path.join(ds, "masks", "frame_000.png"), "w").write("mask")
    open(os.path.join(ds, "sparse", "0", "cameras.txt"), "w").write("PINHOLE")
    open(os.path.join(ds, "_spheresfm_work", "sparse", "0", "cameras.txt"), "w").write("SPHERE")

    staged = splat.stage_clean_dataset(ds)
    assert staged != ds
    # The whole point: Brush must see ONE model, and it must be the PINHOLE one.
    assert not os.path.exists(os.path.join(staged, "_spheresfm_work")), \
        "the SPHERE model must not reach the trainer — that is the Invalid camera model crash"
    assert open(os.path.join(staged, "sparse", "0", "cameras.txt")).read() == "PINHOLE"
    assert sorted(os.listdir(os.path.join(staged, "images"))) == \
        ["frame_000.png", "frame_001.png", "frame_002.png"]
    assert os.path.isfile(os.path.join(staged, "masks", "frame_000.png")), "Brush reads masks/"
    # Hardlinked, not copied: a SplatKit dataset is double-digit GB.
    s = os.stat(os.path.join(staged, "images", "frame_000.png"))
    assert s.st_nlink > 1 or os.name != "nt", "images should be hardlinked where the fs allows"
    print(f"ok  SplatKit dataset staged to one model (nlink={s.st_nlink})")

    # Idempotent: a re-run must not explode on the dir it made last time.
    assert splat.stage_clean_dataset(ds) == staged
    print("ok  staging is re-runnable")
finally:
    shutil.rmtree(root, ignore_errors=True)

# --- the override path must never reach the network, and must reject a bad path.
exe = r"G:\MPI-623-spike\brush\extracted\brush_app.exe"
try:
    splat.ensure_brush(os.path.join(tempfile.gettempdir(), "nope-does-not-exist.exe"))
    raise SystemExit("FAIL: a missing brush_path must raise")
except FileNotFoundError:
    print("ok  a missing brush_path raises instead of silently downloading")

assert splat._platform_key() == "win_amd64"
asset, sha, name = splat.BRUSH_ASSETS["win_amd64"]
assert sha == "b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6"
print("ok  platform key + pinned checksum")

if os.path.isfile(exe):
    assert splat.ensure_brush(exe) == exe
    print("ok  override returns the bench binary, no download")
else:
    print("--  bench binary absent, override-hit case not exercised")


# --- the train() poll loop, with a fake process standing in for Brush. This is the
#     part with no precedent in the pack: progress comes from the export DIRECTORY,
#     because Brush prints nothing off a TTY.
import io

class FakeBrush:
    """Writes export_{iter}.ply on a schedule, prints nothing, then exits."""
    def __init__(self, cmd, stdout=None, stderr=None, exit_code=0, emit=True):
        self.cmd, self.returncode, self._exit_code = cmd, None, exit_code
        self.stdout, self.stderr = io.BytesIO(b""), io.BytesIO(b"boom: invalid camera model")
        out = cmd[cmd.index("--export-path") + 1]
        total = int(cmd[cmd.index("--total-steps") + 1])
        every = int(cmd[cmd.index("--export-every") + 1])
        self._pending = list(range(every, total + 1, every)) if emit else []
        self._out = out
    def poll(self):
        if self._pending:
            step = self._pending.pop(0)
            open(os.path.join(self._out, f"export_{step}.ply"), "w").write("x" * step)
            return None
        self.returncode = self._exit_code
        return self.returncode
    def kill(self): self.returncode = -9
    def wait(self): return self.returncode

node = splat.MpiBrushTrain()
ds = tempfile.mkdtemp(prefix="splattrain-")
os.makedirs(os.path.join(ds, "sparse", "0"))
real_popen, real_sleep = splat.subprocess.Popen, splat.time.sleep
bars = []
splat.comfy.utils.ProgressBar = lambda total: bars.append(_PBar(total)) or bars[-1]
splat.time.sleep = lambda s: None
try:
    splat.subprocess.Popen = lambda cmd, **kw: FakeBrush(cmd)
    (ply,) = node.train(ds, total_steps=30000, export_every=5000, brush_path=exe)
    assert os.path.basename(ply) == "export_30000.ply", f"got {ply}"
    assert bars[-1].n == 30000, f"progress must reach total_steps, got {bars[-1].n}"
    print(f"ok  train() returns the final ply and drove the bar to {bars[-1].n}/30000")

    # A non-zero exit must surface Brush's stderr, not a bare exit code.
    splat.subprocess.Popen = lambda cmd, **kw: FakeBrush(cmd, exit_code=1)
    try:
        node.train(ds, total_steps=10000, export_every=5000, brush_path=exe)
        raise SystemExit("FAIL: a non-zero exit must raise")
    except RuntimeError as e:
        assert "invalid camera model" in str(e), str(e)
        print("ok  a failed bake raises with Brush's own stderr attached")

    # Exit 0 with nothing exported is a silent no-output run — must not be reported
    # as success with a path that does not exist.
    splat.subprocess.Popen = lambda cmd, **kw: FakeBrush(cmd, emit=False)
    try:
        node.train(ds, total_steps=10000, export_every=5000, brush_path=exe)
        raise SystemExit("FAIL: exit 0 with no .ply must raise")
    except RuntimeError as e:
        assert "exported no .ply" in str(e), str(e)
        print("ok  exit 0 with no .ply is an error, not a phantom success")

    # Cancelling the prompt must kill the trainer, not orphan a 45-minute process.
    killed = {}
    # ComfyUI's real InterruptProcessingException derives from BaseException, NOT
    # Exception — an `except Exception:` in the poll loop never fires and Brush is
    # orphaned on the GPU. Model the real base class, not a convenient stand-in.
    class InterruptProcessingException(BaseException):
        pass
    def interrupt(): raise InterruptProcessingException("user cancelled")
    splat.mm.throw_exception_if_processing_interrupted = interrupt
    def spawn(cmd, **kw):
        p = FakeBrush(cmd); killed["p"] = p; return p
    splat.subprocess.Popen = spawn
    try:
        node.train(ds, total_steps=10000, export_every=5000, brush_path=exe)
        raise SystemExit("FAIL: an interrupt must propagate")
    except InterruptProcessingException:
        assert killed["p"].returncode == -9, "the trainer process must be killed"
        print("ok  an interrupted prompt kills the trainer")
finally:
    splat.subprocess.Popen, splat.time.sleep = real_popen, real_sleep
    shutil.rmtree(ds, ignore_errors=True)

print("\nall checks passed")
