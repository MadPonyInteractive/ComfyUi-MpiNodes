"""Tests for switches.py (MpiSwitch and subclasses). No ComfyUI, no GPU.

Run with:
    G:/ComfyUi/python_embeded/python.exe test_switches.py
or any Python 3:
    python test_switches.py
"""
import os
import sys
import types

# Insert repo root so we can import switches.py directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Stub ComfyUI imports ──────────────────────────────────────────────────────

class ExecutionBlocker:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return f"ExecutionBlocker({self.val!r})"


comfy_exec = types.ModuleType("comfy_execution")
comfy_exec_graph = types.ModuleType("comfy_execution.graph")
comfy_exec_graph.ExecutionBlocker = ExecutionBlocker
comfy_exec.graph = comfy_exec_graph
sys.modules["comfy_execution"] = comfy_exec
sys.modules["comfy_execution.graph"] = comfy_exec_graph

# help_funcs needs folder_paths (comfy_paths) and AlwaysEqualProxy/values_equal.
# Stub only the pieces switches.py pulls from it.
folder_paths_stub = types.ModuleType("folder_paths")
folder_paths_stub.get_filename_list = lambda _category: ["None", "a.safetensors", "b.safetensors"]
folder_paths_stub.get_full_path = lambda _category, name: f"/stubs/{name}"
sys.modules["folder_paths"] = folder_paths_stub

# Now import the module under test (it imports as a package member, so patch path).
import importlib
import importlib.util

# Stub help_funcs — switches.py only needs comfy_paths, AlwaysEqualProxy, values_equal.
# Importing the real help_funcs would pull in json.py (local file) which shadows stdlib json.
class AlwaysEqualProxy(str):
    """Matches any type string — mirrors the real AlwaysEqualProxy in help_funcs."""
    def __ne__(self, other):
        return False
    def __eq__(self, other):
        return True

help_funcs_stub = types.ModuleType("help_funcs")
help_funcs_stub.comfy_paths = folder_paths_stub
help_funcs_stub.AlwaysEqualProxy = AlwaysEqualProxy
help_funcs_stub.values_equal = lambda a, b: a == b
sys.modules["help_funcs"] = help_funcs_stub

# Re-wire switches' relative imports before importing it.
_switches_spec = importlib.util.spec_from_file_location(
    "switches",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "switches.py"),
)
_switches = importlib.util.module_from_spec(_switches_spec)
# Patch the loader so `from .help_funcs import ...` resolves correctly.
_switches.__package__ = ""
import builtins as _builtins  # noqa: E402

_real_import = _builtins.__import__


def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 1 and name == "help_funcs":
        return sys.modules["help_funcs"]
    return _real_import(name, globals, locals, fromlist, level)


_builtins.__import__ = _patched_import
try:
    _switches_spec.loader.exec_module(_switches)
finally:
    _builtins.__import__ = _real_import

MpiSwitch = _switches.MpiSwitch
MpiAnySwitch = _switches.MpiAnySwitch
MpiAnySwitch10 = _switches.MpiAnySwitch10
MpiLoraSwitch = _switches.MpiLoraSwitch

# ── Helpers ───────────────────────────────────────────────────────────────────


def is_blocked(val):
    return isinstance(val, ExecutionBlocker)


# ── Tests: MpiAnySwitch (type_name="any", count=5) ───────────────────────────


def test_ungapped_select_1():
    """Ungapped: all five connected. select=1 returns any_1."""
    node = MpiAnySwitch()
    result, idx = node.use_selected(1, any_1="A", any_2="B", any_3="C", any_4="D", any_5="E")
    assert result == "A", result
    assert idx == 1


def test_ungapped_select_3():
    """Ungapped: select=3 returns any_3."""
    node = MpiAnySwitch()
    result, idx = node.use_selected(3, any_1="A", any_2="B", any_3="C", any_4="D", any_5="E")
    assert result == "C", result
    assert idx == 3


def test_ungapped_select_5():
    """Ungapped: select=5 returns any_5."""
    node = MpiAnySwitch()
    result, idx = node.use_selected(5, any_1="A", any_2="B", any_3="C", any_4="D", any_5="E")
    assert result == "E", result
    assert idx == 5


def test_gapped_select_connected_slot():
    """Gapped: any_3 unwired. select=4 must return any_4, not any_3's position."""
    node = MpiAnySwitch()
    # ComfyUI omits any_3 from kwargs because it is not connected.
    result, idx = node.use_selected(4, any_1="A", any_2="B", any_4="D", any_5="E")
    assert result == "D", result
    assert idx == 4


def test_gapped_select_unwired_slot_returns_blocker():
    """Gapped: any_3 unwired. select=3 must return ExecutionBlocker, not any_4."""
    node = MpiAnySwitch()
    result, idx = node.use_selected(3, any_1="A", any_2="B", any_4="D", any_5="E")
    assert is_blocked(result), f"expected ExecutionBlocker, got {result!r}"
    assert idx == 3


def test_gapped_select_last_slot():
    """Gapped: any_2 and any_3 unwired. select=5 returns any_5."""
    node = MpiAnySwitch()
    result, idx = node.use_selected(5, any_1="A", any_4="D", any_5="E")
    assert result == "E", result
    assert idx == 5


def test_out_of_range_returns_blocker():
    """select beyond all connected slots returns ExecutionBlocker."""
    node = MpiAnySwitch()
    result, idx = node.use_selected(6, any_1="A", any_2="B")
    assert is_blocked(result), f"expected ExecutionBlocker, got {result!r}"
    assert idx == 6


def test_no_inputs_returns_blocker():
    """No inputs connected at all returns ExecutionBlocker."""
    node = MpiAnySwitch()
    result, idx = node.use_selected(1)
    assert is_blocked(result), f"expected ExecutionBlocker, got {result!r}"
    assert idx == 1


# ── Tests: check_lazy_status ──────────────────────────────────────────────────


def test_lazy_connected_slot_asks_for_that_name():
    """check_lazy_status for a connected slot returns [name]."""
    node = MpiAnySwitch()
    needed = node.check_lazy_status(3, any_1="A", any_2="B", any_3="C")
    assert needed == ["any_3"], needed


def test_lazy_unconnected_slot_asks_for_nothing():
    """check_lazy_status for an unwired slot returns []."""
    node = MpiAnySwitch()
    needed = node.check_lazy_status(3, any_1="A", any_2="B", any_4="D")
    assert needed == [], needed


def test_lazy_ungapped_all_slots():
    """check_lazy_status for every slot in an ungapped set asks the right name."""
    node = MpiAnySwitch()
    kwargs = {f"any_{i}": chr(64 + i) for i in range(1, 6)}
    for sel in range(1, 6):
        needed = node.check_lazy_status(sel, **kwargs)
        assert needed == [f"any_{sel}"], (sel, needed)


# ── Tests: MpiAnySwitch10 (count=10) ─────────────────────────────────────────


def test_switch10_gapped_select_9():
    """MpiAnySwitch10: any_5 unwired; select=9 returns any_9."""
    node = MpiAnySwitch10()
    kwargs = {f"any_{i}": i * 10 for i in range(1, 11) if i != 5}
    result, idx = node.use_selected(9, **kwargs)
    assert result == 90, result
    assert idx == 9


def test_switch10_gapped_select_unwired():
    """MpiAnySwitch10: any_5 unwired; select=5 returns ExecutionBlocker."""
    node = MpiAnySwitch10()
    kwargs = {f"any_{i}": i * 10 for i in range(1, 11) if i != 5}
    result, idx = node.use_selected(5, **kwargs)
    assert is_blocked(result), f"expected ExecutionBlocker, got {result!r}"
    assert idx == 5


# ── Tests: MpiLoraSwitch (type_name="lora_name", count=5) ────────────────────


def test_lora_switch_ungapped():
    """MpiLoraSwitch: all slots present; select=2 returns lora_name_2."""
    node = MpiLoraSwitch()
    kwargs = {f"lora_name_{i}": f"lora{i}.safetensors" for i in range(1, 6)}
    result, idx = node.use_selected(2, **kwargs)
    assert result == "lora2.safetensors", result
    assert idx == 2


def test_lora_switch_gapped():
    """MpiLoraSwitch: lora_name_2 missing; select=3 returns lora_name_3."""
    node = MpiLoraSwitch()
    kwargs = {f"lora_name_{i}": f"lora{i}.safetensors" for i in range(1, 6) if i != 2}
    result, idx = node.use_selected(3, **kwargs)
    assert result == "lora3.safetensors", result
    assert idx == 3


def test_lora_switch_gapped_unwired_returns_blocker():
    """MpiLoraSwitch: lora_name_2 missing; select=2 returns ExecutionBlocker."""
    node = MpiLoraSwitch()
    kwargs = {f"lora_name_{i}": f"lora{i}.safetensors" for i in range(1, 6) if i != 2}
    result, idx = node.use_selected(2, **kwargs)
    assert is_blocked(result), f"expected ExecutionBlocker, got {result!r}"
    assert idx == 2


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok", name)
    print(f"{len(tests)} passed")
