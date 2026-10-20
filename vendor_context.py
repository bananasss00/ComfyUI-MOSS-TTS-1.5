"""Isolated ("vendored") transformers for MOSS-TTS-ComfyUI.

Why this exists
---------------
ComfyUI's Python environment holds ONE shared `transformers` for every custom
node pack. This pack needs a modern one -- the 8B model's remote code uses
`processing_utils.MODALITY_TO_BASE_CLASS_MAPPING` (a 5.0 name) and its audio
tokenizer configs only parse on >= 5.5.1 -- while other packs in the same
install may still require transformers 4.x. Upgrading the shared copy fixes
MOSS and breaks someone else; staying behind breaks MOSS.

The way out is the same one Higgs_v3-TTS-ComfyUI (branch
`vendored_transformers`) takes: the pack pip-installs its own transformers
into a `vendor/` folder next to these files (see `setup_vendored_dependencies`
in ``__init__.py``), and WHILE a MOSS node executes, that copy temporarily
replaces the shared one in ``sys.modules``. Torch, torchaudio and the Rust
`tokenizers` extension stay shared -- only the pure-Python packages swap.

How the swap works
------------------
Entering :func:`vendored_transformers`:

1. pops every loaded ``transformers`` / ``huggingface_hub`` / ``hf_xet``
   module out of ``sys.modules`` (they are kept in a local dict) -- plus
   ``tokenizers`` / ``safetensors``, but ONLY when the pack has vendored
   copies of those (see ``OPTIONAL_NATIVE_PACKAGES``),
2. re-inserts the modules loaded from ``vendor/`` (kept alive in a module
   cache so their identities are STABLE across node runs -- objects created
   by a previous run, like the cached model bundle, keep checking
   ``isinstance`` against exactly the classes they were built with),
3. puts ``vendor/`` first on ``sys.path`` so fresh imports resolve there,
4. applies small patches the vendored copy needs to coexist with the shared
   environment and with remote model code (details inline).

Leaving it restores the previous state. Native packages are handled
carefully: ``tokenizers`` is a PyO3/Rust extension, and blindly swapping it
would be reckless -- so it (and ``safetensors``) is swapped only when the
pack has vendored its own copy, and even then nothing is ever UNloaded: the
global module object stays alive in ``saved_modules``, while the vendored
.so/.pyd is a second, independent and stateless module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from contextlib import contextmanager

logger = logging.getLogger("MOSS-TTS-ComfyUI")

#: The directory this pack pip-installs its private transformers into.
VENDOR_DIRECTORY_NAME = "vendor"

#: Pure-Python packages that exist twice on disk (vendor/ + site-packages)
#: and are therefore swapped as a set. Order does not matter; the matcher
#: below swaps each of them including every submodule.
VENDOR_PACKAGES = ("transformers", "huggingface_hub", "hf_xet")

#: Compiled (native) packages the vendored transformers needs, which are
#: swapped only when the pack actually vendored a copy (see
#: ``core_dependency_gaps`` / ``setup_vendored_dependencies``). They are NOT
#: swapped blindly: without a vendored copy there is nothing to swap in, and
#: touching the shared copy for nothing would be pointless risk. With one,
#: the swap is the same keep-alive mechanism as for the pure-Python
#: families -- the global module object stays alive in saved_modules, the
#: vendored extension is a second, independent and stateless module.
OPTIONAL_NATIVE_PACKAGES = ("tokenizers", "safetensors")

#: Used when the vendored transformers' own metadata cannot be read (broken
#: install, exotic layout): the last requirement pair this pack was verified
#: against, so the gap install can still do something sensible.
FALLBACK_NATIVE_REQUIREMENTS = {
    "tokenizers": ">=0.23.1,<0.24.0",
    "safetensors": ">=0.8.0",
}

#: Setup completion marker (see ``__init__.py``). Schema 1 (a bare marker
#: file) meant "transformers is in vendor/"; schema 2 adds the native gap
#: step, so an older vendor/ is migrated on the next start without having to
#: delete anything by hand.
VENDOR_READY_MARKER = ".ready"
VENDOR_READY_SCHEMA = 2

#: Set up once, at install time, by ``__init__.py`` /
#: ``setup_vendored_dependencies``: a marker file whose presence means the
#: vendored copy is on disk and complete enough to use. Relative to the
#: vendor/ directory itself (see :func:`has_vendored_transformers`).
VENDOR_MARKER = os.path.join("transformers", "__init__.py")


def vendor_directory() -> str:
    """Absolute path of this pack's vendor/ folder."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), VENDOR_DIRECTORY_NAME)


def has_vendored_transformers() -> bool:
    """True when the isolated transformers is installed in vendor/."""
    return os.path.exists(os.path.join(vendor_directory(), VENDOR_MARKER))


def ready_marker_path() -> str:
    """Absolute path of the setup completion marker."""
    return os.path.join(vendor_directory(), VENDOR_READY_MARKER)


def read_ready_marker() -> dict | None:
    """The parsed completion marker, or None when absent/corrupt."""
    try:
        with open(ready_marker_path(), encoding="utf-8") as marker_file:
            data = json.load(marker_file)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_ready_marker(vendored_gaps: list[str]) -> None:
    """Persist the completion marker; call ONLY after a fully successful setup."""
    with open(ready_marker_path(), "w", encoding="utf-8") as marker_file:
        json.dump(
            {"schema": VENDOR_READY_SCHEMA, "vendored_gaps": list(vendored_gaps)},
            marker_file,
        )


def _swappable_targets(vendor_path: str) -> tuple[str, ...]:
    """The module families this context swaps: the core pure-Python ones
    always, each optional native one only when a vendored copy exists."""
    targets = list(VENDOR_PACKAGES)
    for package in OPTIONAL_NATIVE_PACKAGES:
        if os.path.isdir(os.path.join(vendor_path, package)):
            targets.append(package)
    return tuple(targets)


def _split_requirement(specification: str) -> tuple[str | None, str]:
    """'tokenizers<0.24.0,>=0.23.1' -> ('tokenizers', '<0.24.0,>=0.23.1').
    Tolerates the parenthesised Metadata form and returns ('', spec) misses."""
    match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*)$", specification)
    if match is None:
        return None, ""
    name, rest = match.group(1), match.group(2).strip().strip("(")
    rest = rest.rstrip(")").strip()
    return name, rest


def core_dependency_gaps(vendor_dir: str | None = None, installed_version=None) -> list[str]:
    """Requirements of the VENDORED transformers that the SHARED environment
    does not satisfy -- as pip requirement strings this pack may vendor.

    Only the two native packages are considered (``OPTIONAL_NATIVE_PACKAGES``):
    they are the ones whose version ranges are strict enough to collide
    (transformers 5.x wants tokenizers>=0.23.1,<0.24.0 while a shared
    transformers 4.x wants <=0.23.0 -- no version satisfies both, so the
    shared one must never be upgraded; a private copy in vendor/ is the only
    way out).

    The source of truth is the vendored transformers' own METADATA; the
    comparison uses the shared environment's version via ``importlib.metadata``
    (called OUTSIDE any swap, so it resolves the global copy) + ``packaging``.
    When either is unreadable, the conservative fallback is to vendor the
    package with the requirement from ``FALLBACK_NATIVE_REQUIREMENTS`` --
    pip-installing into vendor/ is idempotent and touches nothing shared.
    """
    vendor_dir = vendor_dir or vendor_directory()
    if installed_version is None:
        from importlib import metadata as _metadata

        def installed_version(name: str):
            try:
                return _metadata.version(name)
            except Exception:
                return None

    # What the vendored transformers demands of the native packages.
    wanted: dict[str, str] = {}
    try:
        import glob

        candidates = sorted(glob.glob(os.path.join(vendor_dir, "transformers-*.dist-info", "METADATA")))
        if candidates:
            with open(candidates[0], encoding="utf-8") as metadata_file:
                for line in metadata_file:
                    if not line.startswith("Requires-Dist:"):
                        continue
                    specification = line.split(":", 1)[1].strip()
                    if ";" in specification:
                        continue  # extra-scoped / platform-marked: not a core dependency
                    name, specifiers = _split_requirement(specification)
                    if name in OPTIONAL_NATIVE_PACKAGES and specifiers:
                        wanted[name] = specifiers
    except Exception:
        wanted = {}

    try:
        from packaging.specifiers import SpecifierSet

        def satisfies(version, specifiers: str) -> bool:
            return version is not None and SpecifierSet(specifiers).contains(version, prereleases=True)
    except Exception:
        satisfies = None  # cannot compare: vendor everything, pip is idempotent

    gaps: list[str] = []
    for package in OPTIONAL_NATIVE_PACKAGES:
        specifiers = wanted.get(package) or FALLBACK_NATIVE_REQUIREMENTS.get(package)
        if not specifiers:
            continue
        version = installed_version(package)
        if satisfies is None or version is None or not satisfies(version, specifiers):
            gaps.append(f"{package}{specifiers}")
    return gaps


def _pack_module_keys(targets) -> list[str]:
    """Every currently-loaded sys.modules key that belongs to ``targets``."""
    return [
        key
        for key in list(sys.modules)
        if any(key == target or key.startswith(target + ".") for target in targets)
    ]


# --------------------------------------------------------------------------
# Patch 1: json serialization of config objects.
#
# transformers config objects (and MOSS's remote-code subclasses of them) are
# dataclass-ish and do not serialize with the stdlib encoder; transformers
# itself never json.dumps them through the STDLIB encoder -- it has its own.
# Any code that does (custom nodes dumping workflow state, ComfyUI metadata,
# our own logging) hits ``TypeError: Object of type ...Config is not JSON
# serializable`` only WHILE the swap is active, because the encoder sees a
# fresh class hierarchy it has no policy for. The patch teaches the stdlib
# encoder the one thing transformers objects need: ``to_dict()``.
# --------------------------------------------------------------------------

_ORIGINAL_JSON_DEFAULT = json.JSONEncoder.default


def _patched_json_default(self, object_to_serialize):
    """Safely serialize nested transformer configs (Qwen3Config & friends)."""
    if hasattr(object_to_serialize, "to_dict"):
        return object_to_serialize.to_dict()
    try:
        return _ORIGINAL_JSON_DEFAULT(self, object_to_serialize)
    except TypeError:
        if hasattr(object_to_serialize, "__dict__"):
            return object_to_serialize.__dict__
        return str(object_to_serialize)


# --------------------------------------------------------------------------
# Patch 2: huggingface_hub's strict type validation.
#
# huggingface_hub validates dataclass-ish fields with isinstance() against
# classes from ITS OWN module. After a swap there are two huggingface_hub
# families alive -- the one an object was built with and the one currently
# first in sys.modules -- and isinstance() between the two fails, rejecting
# perfectly good values. The patch swaps the validator for an always-pass
# stub for the duration of the context and puts the original back afterwards.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Patch 3: transformers 5.x strict processor kwargs vs remote code written
# for transformers 4.x.
#
# transformers 5.x resolves the revision once inside
# AutoProcessor.from_pretrained (kwargs["revision"] = resolve_revision(...))
# and forwards it to the processor class. Stock 5.x processors pop such hub
# kwargs before ProcessorMixin.__init__ sees them, but remote-code processors
# that override from_pretrained -- MOSS's processing_moss_tts.py does exactly
# that -- pass everything through. And 5.x made ProcessorMixin.__init__
# strict: any kwarg that is not one of the processor's declared attributes
# raises ``TypeError: Unexpected keyword argument revision.`` (4.x silently
# setattr'd unknown kwargs, which is what such remote code was written
# against).
#
# The patch restores the 4.x tolerance on the VENDORED copy only: unexpected
# kwargs become plain attributes and the original init runs with a clean
# kwargs dict. The revision itself keeps doing its real job -- pinning the
# commit every file is downloaded from -- it just stops crashing the
# constructor.
# --------------------------------------------------------------------------


def _make_tolerant_processor_init(original_init):
    """Wrap a strict 5.x ``ProcessorMixin.__init__`` with 4.x kwargs tolerance."""

    def tolerant_init(self, *args, **kwargs):
        try:
            allowed = frozenset(self.get_attributes())
        except Exception:
            # A class whose attributes cannot even be computed: degrade to
            # pass-through, the original init raises its own (better) error.
            allowed = frozenset()
        # chat_template and audio_tokenizer are popped by the original init
        # itself -- they must reach it untouched, they are never "extra".
        allowed = allowed | frozenset(("chat_template", "audio_tokenizer"))
        extras = {key: value for key, value in kwargs.items() if key not in allowed}
        if not extras:
            return original_init(self, *args, **kwargs)
        # transformers 4.x accepted unknown processor kwargs by setattr-ing
        # them; remote-code processors were written against exactly that.
        for key, value in extras.items():
            setattr(self, key, value)
        cleaned = {key: value for key, value in kwargs.items() if key in allowed}
        return original_init(self, *args, **cleaned)

    tolerant_init._moss_tts_tolerant_kwargs = True
    return tolerant_init


def _is_vendored_file(file_path, vendor_path: str) -> bool:
    """True when ``file_path`` (a module ``__file__``) lives inside vendor/."""
    if not file_path:
        return False
    vendor_abs = os.path.abspath(vendor_path)
    return os.path.abspath(file_path).startswith(vendor_abs + os.sep)


def _drop_new_transformers_modules(before: frozenset) -> None:
    """Undo a FAILED import attempt: drop every ``transformers.*`` module that
    the attempt added to sys.modules. Without this, a broken/partial vendored
    install (or an empty stand-in directory) would leave half-imported
    modules behind -- and the exit logic would then cache them as if they
    were the vendored family."""
    for key in [
        key
        for key in sys.modules
        if key not in before and (key == "transformers" or key.startswith("transformers."))
    ]:
        sys.modules.pop(key, None)


#: Cache of the modules loaded from vendor/, so their identities survive the
#: gaps between node runs. Objects the pack hands across node boundaries (the
#: MOSS_MODEL bundle) were built from these exact objects.
_VENDORED_MODULES: dict = {}


@contextmanager
def vendored_transformers():
    """Temporarily swap the system transformers for this pack's isolated copy.

    Every code path that touches transformers -- loading a model, and ANY
    node that runs the processor or ``model.generate()`` (their remote code
    lazy-imports transformers from anywhere) -- must run inside this context.
    See nodes.py: the six methods that do are wrapped in it.

    Re-entrant: nesting the context restores the same stable vendored module
    objects and leaves sys.path clean once the outermost block exits.
    """
    vendor_path = vendor_directory()

    if not has_vendored_transformers():
        # No isolated copy on disk (offline first start, failed install,
        # vendor/ deleted): run exactly as the pre-vendoring pack did -- the
        # shared transformers is used as-is. Swapping WITHOUT a vendored copy
        # would not just be pointless: the type-validator patch below would
        # re-import the GLOBAL huggingface_hub mid-swap and the exit logic
        # would then cache that global copy as if it were the vendored one.
        yield
        return

    targets = _swappable_targets(vendor_path)

    saved_modules = {}
    for key in _pack_module_keys(targets):
        saved_modules[key] = sys.modules.pop(key)

    for key, module in _VENDORED_MODULES.items():
        sys.modules[key] = module

    sys.path.insert(0, vendor_path)

    json.JSONEncoder.default = _patched_json_default

    # Patch 3 (see above): make the vendored 5.x ProcessorMixin.__init__
    # tolerant of the kwargs 4.x-era remote code forwards into it. Only the
    # copy FROM vendor/ is patched (the __file__ guard), and the patch is
    # undone on exit, mirroring the type-validator patch below. Nested
    # contexts see the marker and do not wrap twice.
    _orig_processor_init = None
    try:
        _modules_before = frozenset(sys.modules)
        import transformers.processing_utils  # noqa: F401  (resolves vendored first)
        processing_utils = sys.modules.get("transformers.processing_utils")
        if processing_utils is not None and _is_vendored_file(
            getattr(processing_utils, "__file__", None), vendor_path
        ):
            current_init = processing_utils.ProcessorMixin.__init__
            if not getattr(current_init, "_moss_tts_tolerant_kwargs", False):
                _orig_processor_init = current_init
                processing_utils.ProcessorMixin.__init__ = _make_tolerant_processor_init(current_init)
        else:
            # imported, but not the vendored copy: whatever this pulled in
            # must not end up cached as vendored on exit
            _drop_new_transformers_modules(_modules_before)
    except Exception:
        _orig_processor_init = None  # never let the patch break the swap
        _drop_new_transformers_modules(_modules_before)  # failed import: leave no debris

    _orig_type_validator = None
    # Only patch a hub that comes FROM vendor/. The isdir guard matters: a
    # plain `import huggingface_hub` here would fall through to the GLOBAL
    # copy whenever vendor/ lacks the package, and that fresh global import
    # would then be cached as vendored on exit (see the fallback above).
    if os.path.isdir(os.path.join(vendor_path, "huggingface_hub")):
        try:
            import huggingface_hub.dataclasses
            if hasattr(huggingface_hub.dataclasses, "type_validator"):
                _orig_type_validator = huggingface_hub.dataclasses.type_validator
                huggingface_hub.dataclasses.type_validator = lambda *args, **kwargs: None
        except Exception:
            pass

    try:
        yield
    finally:
        if _orig_processor_init is not None:
            try:
                processing_utils = sys.modules.get("transformers.processing_utils")
                if processing_utils is not None and getattr(
                    processing_utils.ProcessorMixin.__init__, "_moss_tts_tolerant_kwargs", False
                ):
                    processing_utils.ProcessorMixin.__init__ = _orig_processor_init
            except Exception:
                pass

        if _orig_type_validator:
            try:
                import huggingface_hub.dataclasses
                huggingface_hub.dataclasses.type_validator = _orig_type_validator
            except Exception:
                pass

        json.JSONEncoder.default = _ORIGINAL_JSON_DEFAULT

        for key in _pack_module_keys(targets):
            _VENDORED_MODULES[key] = sys.modules.pop(key)

        sys.modules.update(saved_modules)

        # remove ONE occurrence -- this context's own. A `while`-style cleanup
        # would also strip the copy an OUTER, still-active context put there
        # and leave fresh imports inside it resolving to the shared packages.
        if vendor_path in sys.path:
            sys.path.remove(vendor_path)
