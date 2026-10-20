"""Pre-installation helper, run by ComfyUI-Manager (and by hand if you like).

Two jobs, in the pack's usual spirit of not touching the shared environment:

1. sanity-check that the heavy ComfyUI-provided packages (torch, torchaudio,
   transformers) exist -- this helper never upgrades or replaces them;
2. pip-install the pack's ISOLATED transformers into ./vendor -- the same
   thing the pack's first in-ComfyUI load does by itself (see
   ``__init__.py::setup_vendored_dependencies`` and ``vendor_context.py``).
   Doing it here just moves the one-time download off the first workflow run.

When the shared environment is too old for transformers 5.x in the two
COMPILED dependencies (tokenizers, safetensors -- e.g. a shared transformers
4.x needs tokenizers<=0.23.0 while ours needs >=0.23.1,<0.24.0), the private
copies are pip-installed into vendor/ as well. Upgrading those in the shared
environment is NOT an option: it would break the shared transformers and
every other pack using it. A ``.ready`` marker records the completed setup,
so re-running this helper never redoes work.

Any other missing package (accelerate, tqdm) is installed into the normal
environment: those are single-version libraries every ComfyUI install
already shares, so there is nothing to isolate.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

CRITICAL_IMPORTS = ["torch", "torchaudio", "transformers"]

# huggingface_hub is not in this list on purpose: it is pip-installed into the
# pack's vendor/ folder together with transformers (see below) and swapped in
# only while a MOSS node runs, so the shared copy never matters.
#
# tokenizers / safetensors are not here EITHER, also on purpose: upgrading
# them in the shared environment can break the shared transformers (their
# version ranges for 4.x and 5.x do not overlap). When the shared copy is too
# old, the pack vendors its own into vendor/ instead (below).
LIGHTWEIGHT_IMPORTS = {
    "accelerate": "accelerate",
    "tqdm": "tqdm",
}

VENDOR_REQUIREMENTS = ["transformers>=5.5.1", "huggingface_hub", "hf_xet"]


def _load_vendor_context():
    """Import the sibling vendor_context.py however this helper was invoked."""
    try:
        import vendor_context

        return vendor_context
    except ImportError:
        pass
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor_context.py")
    spec = importlib.util.spec_from_file_location("vendor_context", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _missing() -> list[str]:
    return [package for module, package in LIGHTWEIGHT_IMPORTS.items() if importlib.util.find_spec(module) is None]


def main() -> int:
    missing_critical = [name for name in CRITICAL_IMPORTS if importlib.util.find_spec(name) is None]
    if missing_critical:
        print(f"Missing critical dependencies: {', '.join(missing_critical)}")
        print("Install this nodepack inside a working ComfyUI environment; this helper will not modify torch, torchaudio, or transformers.")
        return 1

    vendor_context = _load_vendor_context()
    vendor_dir = vendor_context.vendor_directory()
    marker = vendor_context.read_ready_marker()
    setup_complete = (
        vendor_context.has_vendored_transformers()
        and marker
        and marker.get("schema", 0) >= vendor_context.VENDOR_READY_SCHEMA
    )

    if not setup_complete:
        # Also make sure the vendored directory is prepared
        os.makedirs(vendor_dir, exist_ok=True)
        if not vendor_context.has_vendored_transformers():
            print("Installing the isolated transformers (transformers, huggingface_hub, hf_xet) into vendor/ ...")
            subprocess.call([
                sys.executable, "-m", "pip", "install",
                *VENDOR_REQUIREMENTS,
                "--target", vendor_dir, "--upgrade", "--no-deps", "--quiet",
            ])

        gaps = vendor_context.core_dependency_gaps()
        if gaps:
            print(
                "The shared environment is too old for transformers 5.x: vendoring "
                f"{', '.join(gaps)} into vendor/ (nothing shared is upgraded) ..."
            )
            subprocess.call([
                sys.executable, "-m", "pip", "install", *gaps,
                "--target", vendor_dir, "--upgrade", "--no-deps", "--only-binary=:all:", "--quiet",
            ])
        vendor_context.write_ready_marker(gaps)
        print(f"Done. The isolated copy lives in {vendor_dir} and is used only while MOSS-TTS nodes run.")

    missing = _missing()
    if not missing:
        print("ComfyUI-MOSS-TTS-1.5 dependencies are already present.")
        return 0
    print(f"Installing missing lightweight dependencies: {', '.join(missing)}")
    cmd = [sys.executable, "-m", "pip", "install", *missing]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
