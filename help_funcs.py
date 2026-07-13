import hashlib, re, random, os, json, shutil, math, subprocess, ast, operator
import comfy  # type: ignore
import torch  # type:ignore
import folder_paths as comfy_paths  # type: ignore

_lora_cache = {}


# Arithmetic-only expression evaluator. Replaces eval() — the Comfy registry
# bans eval/exec outright (RCE risk), and an eval sandbox is bypassable anyway.
# ast walks the parse tree and executes only whitelisted operators + math
# functions, so an untrusted expression string can't reach arbitrary code.
_MATH_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_MATH_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MATH_FUNCS = {n: getattr(math, n) for n in dir(math) if not n.startswith("__")}


def safe_math(expression, variables):
    """Evaluate an arithmetic `expression` string with the given `variables`
    dict, using only +-*///%** , comparisons, math.* functions and numeric
    literals. Raises ValueError on anything else (attribute access, calls to
    non-math names, imports, etc.)."""
    _CMP = {
        ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
        ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    }

    def _ev(node):
        if isinstance(node, ast.Constant):  # numbers (and bools)
            return node.value
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ValueError(f"unknown name '{node.id}'")
        if isinstance(node, ast.BinOp) and type(node.op) in _MATH_BINOPS:
            return _MATH_BINOPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _MATH_UNARYOPS:
            return _MATH_UNARYOPS[type(node.op)](_ev(node.operand))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            return _CMP[type(node.ops[0])](_ev(node.left), _ev(node.comparators[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _MATH_FUNCS.get(node.func.id)
            if fn is None or node.keywords:
                raise ValueError(f"disallowed call '{getattr(node.func, 'id', '?')}'")
            return fn(*[_ev(a) for a in node.args])
        raise ValueError(f"disallowed expression: {ast.dump(node)}")

    return _ev(ast.parse(expression, mode="eval").body)


def is_empty_value(value):
    """True if value is 'empty' for any ComfyUI type.

    Empty: None, empty string/list/tuple/dict, zero-element tensor
    (IMAGE/MASK/LATENT with 0 batch), AUDIO with an empty waveform.
    Not empty: 0, 0.0, False, and any populated value.
    """
    # ponytail: falsiness would wrongly treat 0/0.0/False as empty.
    if value is None:
        return True
    # tensors: empty = zero elements. 0-d scalar tensors have numel 1 (len() would crash).
    if torch.is_tensor(value):
        return value.numel() == 0
    # AUDIO is {"waveform": tensor, "sample_rate": int}; empty = empty waveform.
    if isinstance(value, dict):
        wf = value.get("waveform")
        if torch.is_tensor(wf):
            return wf.numel() == 0
        return len(value) == 0
    if hasattr(value, "__len__"):
        return len(value) == 0
    return False


def round_to_multiple(value, multiple_of, round):
    if multiple_of <= 0:
        return (value,)  # avoid division by zero

    if round:
        rounded = math.ceil(value / multiple_of) * multiple_of
    else:
        rounded = math.floor(value / multiple_of) * multiple_of

    return rounded


def crop_offset(outer, inner, position, axis):
    """Top-left offset to place an `inner`-sized crop inside `outer` along one
    axis, given a position anchor. axis 'x' honors left/right, axis 'y' honors
    top/bottom; center (and the irrelevant axis) centers."""
    if position == "left" and axis == "x":
        return 0
    if position == "right" and axis == "x":
        return outer - inner
    if position == "top" and axis == "y":
        return 0
    if position == "bottom" and axis == "y":
        return outer - inner
    return (outer - inner) // 2


def create_mask_from_bbox(
    image_tensor, bbox, normalized=False, bbox_format="xywh"
):
    if len(image_tensor.shape) == 4:
        B, H, W, _ = image_tensor.shape
    else:
        B, H, W = image_tensor.shape

    device = image_tensor.device
    dtype = (
        image_tensor.dtype
        if image_tensor.dtype in [torch.float32, torch.float16]
        else torch.float32
    )

    if bbox_format == "xywh":
        x_min, y_min, box_w, box_h = bbox
        if normalized:
            x_min *= W
            y_min *= H
            box_w *= W
            box_h *= H
        x_max = x_min + box_w
        y_max = y_min + box_h
    elif bbox_format == "xyxy":
        x_min, y_min, x_max, y_max = bbox
        if normalized:
            x_min *= W
            y_min *= H
            x_max *= W
            y_max *= H
    else:
        raise ValueError(f"Unsupported bbox_format: {bbox_format}")

    x_min = int(max(0, min(x_min, W)))
    y_min = int(max(0, min(y_min, H)))
    x_max = int(max(x_min, min(x_max, W)))
    y_max = int(max(y_min, min(y_max, H)))

    mask = torch.zeros((B, H, W), dtype=dtype, device=device)
    mask[:, y_min:y_max, x_min:x_max] = 1.0

    return mask  # shape: [B, H, W]


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


def copy_folder(src: str, dst: str):
    """
    Recursively copies all contents from src folder to dst folder.
    Creates dst if it doesn't exist.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def write_lines_to_file(path: str, lines: list[str]) -> None:
    """
    Overwrites the file at `path` with each string from `lines` on its own line.
    """
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def build_path(folder, filename):
    """Join folder and filename and return full path."""
    return os.path.join(folder, filename)


def read_lines(file_path):
    """Read a text file and return a list of cleaned lines."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def list_text_files(folder_path):
    """Return a list of .txt filenames in the given folder."""
    return [
        filename
        for filename in os.listdir(folder_path)
        if filename.endswith(".txt")
        and os.path.isfile(os.path.join(folder_path, filename))
    ]


def load_json_dict_abs(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[MpiNodes] Error loading JSON from {path}: {e}")
        return {}


def save_json_dict_abs(data, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[MpiNodes] Error saving JSON to {path}: {e}")


def load_json_dict(filename):
    """
    data = load_json_dict("user/presets.json")
    print(data.get("preset_1", {}))
    """
    path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[MpiNodes] Error loading {filename}: {e}")
        return {}


def save_json_dict(data, filename):
    """
    my_data = {
        "preset_1": {
            "choices": ["apple", "banana", "cherry"]
        }
    }

    save_json_dict(my_data, "user/presets.json")
    """
    path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[MpiNodes] Error saving {filename}: {e}")


def aspect_ratio(width, height):
    ratio = width / height
    standard_ratios = {
        "1:1": 1.0,
        "4:3": 4 / 3,
        "3:2": 3 / 2,
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "2:3": 2 / 3,
        "3:4": 3 / 4,
    }
    closest = min(standard_ratios.items(), key=lambda x: abs(ratio - x[1]))
    return closest[0]


def load_lora_cached(lora_name):
    if lora_name == "None":
        return None

    lora_path = comfy_paths.get_full_path("loras", lora_name)
    if lora_path in _lora_cache:
        return _lora_cache[lora_path]

    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    _lora_cache[lora_path] = lora
    return lora


def load_combo(file_name):
    path = os.path.join(os.path.dirname(__file__), "user/combos", file_name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return tuple(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"[MpiComboLoader] Error loading {file_name}: {e}")
        return ("None",)


def wrap_in_underscores(s):
    # Remove any leading/trailing underscores
    cleaned = re.sub(r"^_*(.*?)_*$", r"\1", s.strip())
    cleaned = cleaned.replace(" ", "_")
    # Wrap the cleaned string in underscores
    return f"_{cleaned}_" if cleaned else ""


def pick_seeded_item(items, seed):
    # Picks an item from a list
    random.seed(seed)
    return random.choice(items)


def seeded_shuffle(items: list, seed: int) -> list:
    def hash_key(item, index):
        key = f"{seed}-{index}-{item}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:8], 16)  # Use first 8 hex digits for sorting

    return sorted(items, key=lambda item: hash_key(item, items.index(item)))


def probabilistic_shuffle(items: list, randomness: float, seed: int) -> list:
    assert 0.0 <= randomness <= 1.0, "randomness must be between 0.0 and 1.0"

    def hash_item(item, index):
        # Combine seed, index, and item into a unique string
        key = f"{seed}-{index}-{item}"
        # Create a deterministic hash
        h = hashlib.sha256(key.encode()).hexdigest()
        # Convert hash to float between 0.0 and 1.0
        return int(h[:8], 16) / 0xFFFFFFFF

    # Generate a score for each item based on its hash and randomness
    scored_items = [
        (item, hash_item(item, i) * randomness + i * (1 - randomness))
        for i, item in enumerate(items)
    ]

    # Sort by score
    scored_items.sort(key=lambda x: x[1])

    # Return reordered list
    return [item for item, _ in scored_items]


def get_random_bool(probability: float, seed: int):
    # Hash the seed to get a pseudo-random number
    hashed = (seed * 2654435761) & 0xFFFFFFFF
    # Normalize to [0.0, 1.0]
    rand_value = hashed / 0xFFFFFFFF
    # Compare against the probability threshold
    return rand_value < probability


def get_blacklist(prompt_list):
    blacklist = [
        word.strip()
        for entry in prompt_list
        for block in entry["blocks"]
        if block.strip()  # skip empty or whitespace-only blocks
        for word in block.split(",")
        if word.strip()  # skip empty words after splitting
    ]
    return sorted(set(blacklist))


def merge_unique_by_key(list1, list2, key="title"):
    # Merge 2 dicts based on a unique key
    seen_titles = {d[key] for d in list1}
    combined = list1 + [d for d in list2 if d[key] not in seen_titles]
    return combined


def resolve_video_path(path):
    """Resolve a video path string the way VHS_LoadVideoPath does.

    The incoming path may be an absolute path (local runs) or a bare basename
    (remote engines upload to ComfyUI's input dir and inject only the filename).
    Returns the first existing resolution, falling back to the raw path.
    """
    if not path:
        return path

    if os.path.isabs(path) and os.path.exists(path):
        return path

    # Mirror VHS resolution: try the annotated-filepath helper, then the
    # plain input directory.
    try:
        annotated = comfy_paths.get_annotated_filepath(path)
        if annotated and os.path.exists(annotated):
            return annotated
    except Exception:
        pass

    candidate = os.path.join(comfy_paths.get_input_directory(), path)
    if os.path.exists(candidate):
        return candidate

    return path


def find_ffmpeg():
    """Locate an ffmpeg executable, mirroring VHS_LoadVideoPath's resolution.

    ComfyUI portable does NOT ship ffprobe and rarely has either binary on
    PATH; what it does have is the ffmpeg bundled by imageio-ffmpeg (the same
    one VHS uses). So detect audio with ffmpeg, not ffprobe.

    Order: VHS_FORCE_FFMPEG_PATH env -> imageio-ffmpeg bundle -> PATH. Returns
    None if nothing is found.
    """
    forced = os.environ.get("VHS_FORCE_FFMPEG_PATH")
    if forced:
        return forced

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    return shutil.which("ffmpeg")


def video_has_audio_stream(path):
    """Return True if the video file at `path` contains an audio stream.

    Uses ffmpeg (inspect-only here, encoder/GPU-agnostic) to read container
    metadata. ffmpeg with no output target errors out but prints the input's
    stream layout to stderr; an audio track shows up as a "Stream ... Audio:"
    line. Detection reads the FILE, not an AUDIO object, because lazy audio
    maps report not-None even when no audio track exists.

    Defaults to False if the path is missing, ffmpeg is unavailable, or the
    probe errors — safer to produce video-only output than to crash a workflow
    on a phantom audio input.
    """
    resolved = resolve_video_path(path)
    if not resolved or not os.path.exists(resolved):
        return False

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[MpiNodes] no ffmpeg found; cannot check for audio")
        return False

    try:
        # `-i <file>` with no output makes ffmpeg dump stream info to stderr
        # and exit non-zero ("At least one output file must be specified") —
        # expected, we only want the printed stream layout.
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", resolved],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return "Audio:" in result.stderr
    except Exception as e:
        print(f"[MpiNodes] ffmpeg audio check failed for {resolved}: {e}")
        return False
