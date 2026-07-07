"""
Server routes registered at ComfyUI startup (imported by __init__.py).

These are NOT workflow nodes — they never appear on the canvas. They are plain
aiohttp HTTP handlers attached to ComfyUI's PromptServer, alongside the core
/prompt, /object_info, etc. routes.

    POST /mpi/reload-extra-paths  — re-read extra_model_paths.yaml at RUNTIME so a
    model folder added mid-session becomes visible to /prompt validation WITHOUT
    restarting ComfyUI. (Cubric Vision MPI-219.)
"""
import logging

from server import PromptServer  # type: ignore
from aiohttp import web


@PromptServer.instance.routes.post("/mpi/reload-extra-paths")
async def mpi_reload_extra_paths(request):
    """
    Re-invoke the same yaml loader ComfyUI runs at boot. This re-registers every
    search path in the yaml via folder_paths.add_model_folder_path. The filename
    list cache self-heals: get_filename_list() sees the newly-registered folder
    absent from its cached snapshot and rescans on the next call, so no explicit
    cache invalidation is needed.

    Body: { "yaml_path": "<absolute path to extra_model_paths.yaml>" }
    """
    from utils.extra_config import load_extra_path_config  # type: ignore

    try:
        data = await request.json()
    except Exception:
        data = {}
    yaml_path = (data or {}).get("yaml_path")
    if not yaml_path:
        return web.json_response({"ok": False, "error": "yaml_path required"}, status=400)

    try:
        load_extra_path_config(yaml_path)
        logging.info("[MpiNodes] reloaded extra model paths from %s", yaml_path)
        return web.json_response({"ok": True, "yaml_path": yaml_path})
    except FileNotFoundError:
        return web.json_response({"ok": False, "error": "yaml not found", "yaml_path": yaml_path}, status=404)
    except Exception as err:  # noqa: BLE001 — surface any yaml/registration failure to caller
        logging.error("[MpiNodes] reload-extra-paths failed: %s", err)
        return web.json_response({"ok": False, "error": str(err)}, status=500)
