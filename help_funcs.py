import hashlib, re, random, os, json, shutil
import comfy  # type: ignore
import folder_paths as comfy_paths  # type: ignore

_lora_cache = {}


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
