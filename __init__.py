"""MOSS-TTS-ComfyUI — OpenMOSS MOSS-TTS v1.5 (1.7B Local-Transformer + 8B full model).

TTS, zero-shot voice cloning, audio continuation and duration control.

This pack runs against its OWN isolated ("vendored") transformers: on first
load it pip-installs a private transformers >= 5.5.1 into ./vendor next to
this file, and while a MOSS node executes that copy temporarily replaces the
ComfyUI-wide one in sys.modules (vendor_context.py). The shared transformers —
and every other pack depending on it — is untouched. The 8B model needs >= 5.5.1;
the Local-Transformer tolerates 4.x, so if the vendored install ever fails
(offline first start, no pip), the pack still works with whatever the shared
environment has, exactly as it did before vendoring.

NODE_CLASS_MAPPINGS is re-exported at module top level as a plain import so the
ComfyUI Registry's static (AST) node parser can discover all ten nodes. Do NOT
wrap this import in a try/except with an empty-dict fallback — the parser then
records the empty dict and the pack shows "No nodes found". ComfyUI's own custom-
node loader already catches and logs import errors gracefully, so no guard here.
The vendored-dependency setup below is therefore allowed to fail on its own —
it logs and moves on, but never raises.
"""

import logging
import os
import subprocess
import sys

logger = logging.getLogger("MOSS-TTS-ComfyUI")


def setup_vendored_dependencies() -> None:
    """Pip-install an isolated transformers into ./vendor, once.

    Everything the swap needs is pure Python (transformers, huggingface_hub,
    hf_xet) and is installed with --no-deps --target: nothing lands in the
    ComfyUI-wide site-packages, and the shared environment never changes.
    Dependencies those packages would pull (requests, tqdm, pyyaml, ...) are
    NOT vendored on purpose -- they are single-version, conflict-free and
    already ship with ComfyUI.

    The two compiled dependencies (tokenizers, safetensors) are special: the
    shared environment may be too old for transformers 5.x in exactly those
    (a shared transformers 4.x needs tokenizers<=0.23.0, ours needs
    >=0.23.1,<0.24.0 -- no version satisfies both). When the shared copy does
    not satisfy the vendored transformers' own requirement, a private copy is
    pip-installed into vendor/ as well -- instead of upgrading anything
    shared (see ``vendor_context.core_dependency_gaps``). A ``.ready`` marker
    records the completed setup (schema 2), so an older vendor/ is migrated
    on the next start without having to delete anything by hand.

    Skipped entirely when a previous run completed the whole setup, so an
    offline restart after a successful install costs nothing. On failure the
    pack falls back to the shared transformers: the Local-Transformer keeps
    working there, and the 8B raises the loader's usual "upgrade transformers"
    message only if the shared one is too old for it.
    """
    try:
        try:
            from . import vendor_context  # normal: pack imported as a package
        except ImportError:
            import vendor_context  # script / by-path context

        vendor_dir = vendor_context.vendor_directory()
        marker = vendor_context.read_ready_marker()
        if (
            vendor_context.has_vendored_transformers()
            and marker
            and marker.get("schema", 0) >= vendor_context.VENDOR_READY_SCHEMA
        ):
            return  # a previous run completed everything, including the gap step

        if not vendor_context.has_vendored_transformers():
            print(
                "[MOSS-TTS] isolated transformers is missing or incomplete — (re)installing it into "
                f"{vendor_dir} (one-time, does not touch your ComfyUI environment) ..."
            )
            os.makedirs(vendor_dir, exist_ok=True)
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "transformers>=5.5.1",
                    "huggingface_hub",
                    "hf_xet",
                    "--target", vendor_dir,
                    "--upgrade",
                    "--no-deps",
                    "--quiet",
                ],
                check=True,
            )

        # The shared environment may be too old for transformers 5.x in the
        # two COMPILED pieces that cannot be blindly swapped (tokenizers,
        # safetensors). Vendor the gaps -- never upgrade anything shared.
        gaps = vendor_context.core_dependency_gaps()
        if gaps:
            print(
                "[MOSS-TTS] the shared environment is too old for transformers 5.x: vendoring "
                f"{', '.join(gaps)} into vendor/ instead of upgrading anything shared ..."
            )
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install", *gaps,
                    "--target", vendor_dir,
                    "--upgrade",
                    "--no-deps",
                    "--only-binary=:all:",
                    "--quiet",
                ],
                check=True,
            )

        vendor_context.write_ready_marker(gaps)
        print("[MOSS-TTS] isolated transformers installed into vendor/.")
    except Exception as exception:  # noqa: BLE001 — startup must never die here
        message = (
            f"[MOSS-TTS] could not set up the isolated transformers "
            f"({exception}). Falling back to the ComfyUI-wide transformers; "
            f"the 8B model needs >= 5.5.1 there. Re-run once with network "
            f"access to retry the isolated install."
        )
        try:
            logger.warning(message)
        except Exception:
            pass
        print(message)


setup_vendored_dependencies()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
