from pathlib import Path

# Keep the public module namespace unchanged while storing implementation files
# below h3_t8. Search it first so leftovers from an older install cannot win.
_package_root = Path(__file__).resolve().parent
_runtime_root = _package_root / "h3_t8"

if __package__:
    __path__ = [str(_runtime_root), str(_package_root)]
    from .nodes import comfy_entrypoint
else:  # Allows direct test collection from a hyphenated custom-node directory.
    import importlib.util
    import sys
    import types

    _package_name = "_minimax_h3_audio_t8_direct"
    _package = types.ModuleType(_package_name)
    _package.__path__ = [str(_runtime_root), str(_package_root)]
    sys.modules.setdefault(_package_name, _package)
    _spec = importlib.util.spec_from_file_location(f"{_package_name}.nodes", _runtime_root / "nodes.py")
    _nodes = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _nodes
    assert _spec.loader is not None
    _spec.loader.exec_module(_nodes)
    comfy_entrypoint = _nodes.comfy_entrypoint


WEB_DIRECTORY = "./web"


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
