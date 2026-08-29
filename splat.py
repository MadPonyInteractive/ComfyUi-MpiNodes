"""Gaussian-splat training nodes.

`MpiBrushTrain` shells out to Brush (github.com/ArthurBrussee/brush, Apache-2.0)
to turn a COLMAP dataset into a trained `.ply`. Brush is a native binary, fetched
per-platform on first use and checksum-verified — the same shape SplatKit already
uses for its `colmap_sphere` SfM binary.

Two runtime facts drive the odd bits of this file, both measured rather than
assumed (Cubric Vision MPI-623, Phase 0):

- **Brush writes zero bytes to stdout when it is not attached to a TTY.** There is
  no `N/M Steps` line to parse. Progress is read from the export directory
  instead: Brush drops `export_{iter}.ply` every `--export-every` steps, and the
  filename carries the step count. Silence is normal and must never be read as a
  failure.
- **A SplatKit dataset holds four COLMAP models**, two of them under
  `_spheresfm_work/` using camera model 11 (SPHERE). Brush picks between them
  nondeterministically and dies with `Invalid camera model` on the unlucky runs,
  so the trainer is handed a root containing exactly one model.
"""

import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import zipfile

import folder_paths  # type: ignore
import comfy.utils  # type: ignore
import comfy.model_management as mm  # type: ignore

from .help_funcs import sha256_file


BRUSH_VERSION = "v0.3.0"
BRUSH_RELEASE_URL = "https://github.com/ArthurBrussee/brush/releases/download"

# asset, sha256, binary name inside the archive. These are the checksums published
# beside each asset; the Windows one was re-verified against a downloaded copy.
# Pinned here rather than fetched at runtime — a checksum served from the same host
# as the payload proves nothing about that host.
BRUSH_ASSETS = {
    "win_amd64": (
        "brush-app-x86_64-pc-windows-msvc.zip",
        "b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6",
        "brush_app.exe",
    ),
    "macos_arm64": (
        "brush-app-aarch64-apple-darwin.tar.xz",
        "65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c",
        "brush_app",
    ),
    "linux_amd64": (
        "brush-app-x86_64-unknown-linux-gnu.tar.xz",
        "4f0f9a8785d1951c62df26aae247c02c5bba32b00f40b06df4e1c9b867399e20",
        "brush_app",
    ),
}


def _platform_key():
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows" and machine in ("amd64", "x86_64"):
        return "win_amd64"
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "macos_arm64"
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return "linux_amd64"
    return None


def _brush_dir():
    return os.path.join(os.path.dirname(__file__), "bin", f"brush-{BRUSH_VERSION}")


def _find_binary(root, name):
    # Brush 0.3.0 archives are flat, but do not bet the node on that staying true.
    direct = os.path.join(root, name)
    if os.path.isfile(direct):
        return direct
    for dirpath, _dirnames, filenames in os.walk(root):
        if name in filenames:
            return os.path.join(dirpath, name)
    return None


def ensure_brush(override=""):
    """Return a path to the Brush binary, downloading and verifying it once.

    `override` short-circuits everything, so an installer that manages the binary
    itself — or a bench with a copy already on disk — never reaches the network.
    """
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"brush_path does not exist: {override}")
        return override

    key = _platform_key()
    if key is None:
        raise RuntimeError(
            f"Brush publishes no prebuilt binary for {platform.system()} "
            f"{platform.machine()}. Build it from source and pass brush_path."
        )

    asset, expected_sha, exe_name = BRUSH_ASSETS[key]
    target_dir = _brush_dir()
    existing = _find_binary(target_dir, exe_name)
    if existing:
        return existing

    os.makedirs(target_dir, exist_ok=True)
    archive = os.path.join(target_dir, asset)
    url = f"{BRUSH_RELEASE_URL}/{BRUSH_VERSION}/{asset}"
    print(f"[MpiNodes] downloading Brush {BRUSH_VERSION} ({asset}) — this happens once")
    urllib.request.urlretrieve(url, archive)

    actual_sha = sha256_file(archive)
    if actual_sha != expected_sha:
        os.remove(archive)
        raise RuntimeError(
            f"Brush checksum mismatch for {asset}: expected {expected_sha}, "
            f"got {actual_sha}. The download was discarded."
        )

    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(target_dir)
    os.remove(archive)

    binary = _find_binary(target_dir, exe_name)
    if not binary:
        raise RuntimeError(f"{asset} did not contain {exe_name}")
    if os.name != "nt":
        os.chmod(binary, 0o755)
    # Apache-2.0 §4(a): the licence travels with the binary. Brush 0.3.0 ships no
    # NOTICE file, so extracting LICENSE alongside discharges the obligation.
    if not os.path.isfile(os.path.join(target_dir, "LICENSE")):
        print(f"[MpiNodes] warning: Brush LICENSE not found in {target_dir}")
    return binary


def stage_clean_dataset(dataset_path):
    """Give Brush a root holding exactly ONE COLMAP model.

    Returns the root to train against: `dataset_path` itself when it is already
    unambiguous, otherwise a staged subdirectory. Images are hardlinked where the
    filesystem allows — a SplatKit dataset runs to double-digit GB, and copying it
    to work around a model-picking bug is not a fair trade.
    """
    if not os.path.isdir(os.path.join(dataset_path, "_spheresfm_work")):
        return dataset_path

    images = os.path.join(dataset_path, "images")
    sparse0 = os.path.join(dataset_path, "sparse", "0")
    if not os.path.isdir(images) or not os.path.isdir(sparse0):
        raise FileNotFoundError(
            f"{dataset_path} carries _spheresfm_work/ but no images/ + sparse/0 to stage from"
        )

    clean = os.path.join(dataset_path, "_mpi_clean")
    clean_images = os.path.join(clean, "images")
    os.makedirs(clean_images, exist_ok=True)
    shutil.copytree(sparse0, os.path.join(clean, "sparse", "0"), dirs_exist_ok=True)

    linked = 0
    for name in sorted(os.listdir(images)):
        src = os.path.join(images, name)
        dst = os.path.join(clean_images, name)
        if not os.path.isfile(src) or os.path.exists(dst):
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        linked += 1

    # Brush picks per-view masks up automatically when the folder is present.
    masks = os.path.join(dataset_path, "masks")
    if os.path.isdir(masks):
        shutil.copytree(masks, os.path.join(clean, "masks"), dirs_exist_ok=True)

    print(f"[MpiNodes] staged {linked} images into a single-model root: {clean}")
    return clean


def exported_step(filename):
    """`export_15000.ply` -> 15000. Anything else -> None."""
    stem = os.path.splitext(filename)[0]
    _, _, tail = stem.rpartition("_")
    return int(tail) if tail.isdigit() else None


class MpiBrushTrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {"default": "", "multiline": False}),
                "total_steps": ("INT", {"default": 30000, "min": 100, "max": 200000, "step": 100}),
            },
            "optional": {
                "export_every": ("INT", {"default": 5000, "min": 100, "max": 200000, "step": 100}),
                "sh_degree": ("INT", {"default": 3, "min": 0, "max": 3}),
                "max_splats": ("INT", {"default": 10000000, "min": 10000, "max": 100000000, "step": 10000}),
                "brush_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "train"
    CATEGORY = "MpiNodes/Splat"
    DESCRIPTION = (
        "Trains a Gaussian splat from a COLMAP dataset with Brush and returns the "
        "path of the exported .ply. Brush is downloaded and checksum-verified on "
        "first use; set brush_path to use a copy you manage yourself. A 30000-step "
        "bake takes tens of minutes — a scene is a durable asset, not a generation."
    )

    def train(self, dataset_path, total_steps, export_every=5000, sh_degree=3,
              max_splats=10000000, brush_path=""):
        dataset_path = (dataset_path or "").strip().strip('"')
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"dataset_path is not a directory: {dataset_path}")

        binary = ensure_brush((brush_path or "").strip().strip('"'))
        train_root = stage_clean_dataset(dataset_path)

        # A UNIQUE export dir, not a timestamped guess. The dir is also the progress
        # signal and the completion check, so a collision with a previous bake would
        # let stale `export_*.ply` files read as this run's output — including making
        # a run that exported nothing at all look like a success.
        stem = os.path.basename(os.path.normpath(dataset_path))
        splat_root = os.path.join(folder_paths.get_output_directory(), "splats")
        os.makedirs(splat_root, exist_ok=True)
        out_dir = tempfile.mkdtemp(prefix=f"{stem}_", dir=splat_root)

        cmd = [
            binary, train_root,
            "--total-steps", str(total_steps),
            "--export-path", out_dir,
            "--export-every", str(export_every),
            "--sh-degree", str(sh_degree),
            "--max-splats", str(max_splats),
        ]
        print(f"[MpiNodes] brush: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        pbar = comfy.utils.ProgressBar(total_steps)
        seen = 0
        try:
            while proc.poll() is None:
                try:
                    mm.throw_exception_if_processing_interrupted()
                except BaseException:
                    # BaseException, not Exception: ComfyUI's InterruptProcessingException
                    # derives from BaseException, so `except Exception` never fires and
                    # a cancelled prompt leaves Brush running on the GPU for another
                    # 45 minutes with nothing left to collect it. KeyboardInterrupt on
                    # the ComfyUI process itself lands here for the same reason.
                    proc.kill()
                    proc.wait()
                    raise
                # Brush prints nothing off a TTY, so the export directory IS the
                # progress signal. A silent run is a working run.
                steps = [exported_step(f) for f in os.listdir(out_dir) if f.endswith(".ply")]
                latest = max([s for s in steps if s is not None], default=0)
                if latest > seen:
                    pbar.update(latest - seen)
                    seen = latest
                time.sleep(2)

            if proc.returncode != 0:
                stderr = (proc.stderr.read() or b"").decode("utf-8", "replace").strip()
                raise RuntimeError(
                    f"Brush exited {proc.returncode}. Last output:\n{stderr[-2000:]}"
                )
        finally:
            for stream in (proc.stdout, proc.stderr):
                if stream and not stream.closed:
                    stream.close()

        plies = [f for f in os.listdir(out_dir) if f.endswith(".ply")]
        if not plies:
            raise RuntimeError(f"Brush exited 0 but exported no .ply into {out_dir}")
        final = max(plies, key=lambda f: exported_step(f) or 0)
        ply_path = os.path.join(out_dir, final)
        print(f"[MpiNodes] brush finished: {ply_path} "
              f"({os.path.getsize(ply_path) / (1024 * 1024):.0f} MB)")
        return (ply_path,)
