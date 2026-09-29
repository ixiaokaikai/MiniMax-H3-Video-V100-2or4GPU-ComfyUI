"""Backward-compatible command-line entry; implementation lives in h3_t8."""
from pathlib import Path
import runpy
import importlib.util
import sys

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "h3_t8/trt_vae_prepare_flex.py"), run_name="__main__")
else:
    # Preserve top-level imports of the historical command modules as well.
    _spec = importlib.util.spec_from_file_location(__name__, Path(__file__).parent / "h3_t8/trt_vae_prepare_flex.py")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
