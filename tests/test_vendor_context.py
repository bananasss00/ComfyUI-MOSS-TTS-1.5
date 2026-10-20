"""Self-check for the isolated ("vendored") transformers swap (vendor_context.py).

The swap is the 0.6.5 mechanism that lets this pack run its own transformers
while the rest of ComfyUI keeps the shared one. It is easy to get subtly wrong
(a module left in sys.modules, a sys.path entry that never comes off, a JSON
patch that stays after the node finished), and every one of those bugs looks
like a different pack's crash, hours later. So this check pins the mechanics
down with FAKE transformers/huggingface_hub modules -- nothing here needs the
real libraries, and none of the real ones is loaded or modified.

Plain asserts, no pytest (see test_model_paths.py for why). Run:

    python tests/test_vendor_context.py

Needs nothing but the stdlib.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SWAPPED = ("transformers", "huggingface_hub", "hf_xet")


def _import_vendor_context():
    spec = importlib.util.spec_from_file_location("vendor_context_selfcheck", ROOT / "vendor_context.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_module(name: str, marker: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__fake_marker__ = marker  # identity that survives pickling into asserts
    return mod


def _install_fake_global_modules():
    """Pretend the shared environment has transformers already loaded."""
    saved = {}
    for name in SWAPPED:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = _fake_module(name, "global")
        sys.modules[name + ".sub"] = _fake_module(name + ".sub", "global")
    return saved


def _restore_fake_global_modules(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    for name in SWAPPED:
        sys.modules.pop(name + ".sub", None)


class _ConfigLike:
    """Stand-in for a transformers config object: only .to_dict() works."""

    def to_dict(self) -> dict:
        return {"model_type": "fake"}


def _make_fake_vendor(tmp_vendor: str) -> None:
    """Give the fake vendor/ the one marker the swap gates on.

    Deliberately NO huggingface_hub/ directory: with it absent, the
    type-validator patch is skipped (its isdir guard) and nothing reaches
    into site-packages mid-test.
    """
    os.makedirs(os.path.join(tmp_vendor, "transformers"), exist_ok=True)
    Path(tmp_vendor, "transformers", "__init__.py").write_text("", encoding="utf-8")


def test_fallback_without_vendor_is_a_noop(vc, tmp_vendor: str):
    """No vendored copy on disk -> nothing is swapped, nothing is cached.

    This is the offline-first-start / failed-install degradation, and the
    guard matters: a swap without a vendored copy would re-import the GLOBAL
    huggingface_hub mid-context and cache it as if it were vendored.
    """
    saved = _install_fake_global_modules()
    try:
        original_default = json.JSONEncoder.default
        with vc.vendored_transformers():
            assert sys.modules["transformers"].__fake_marker__ == "global"
            assert tmp_vendor not in sys.path
            assert json.JSONEncoder.default is original_default
        assert sys.modules["transformers"].__fake_marker__ == "global"
        assert vc._VENDORED_MODULES == {}
    finally:
        json.JSONEncoder.default = original_default
        _restore_fake_global_modules(saved)


def test_swap_and_restore(vc, tmp_vendor: str):
    _make_fake_vendor(tmp_vendor)
    saved = _install_fake_global_modules()
    try:
        with vc.vendored_transformers():
            # the global family is out of the way -- a fresh `import transformers`
            # would now resolve through sys.path, where vendor/ comes first
            assert "transformers" not in sys.modules
            assert "transformers.sub" not in sys.modules
            assert sys.path[0] == tmp_vendor

        # ... and the shared environment is exactly as it was
        assert sys.modules["transformers"].__fake_marker__ == "global"
        assert sys.modules["transformers.sub"].__fake_marker__ == "global"
        assert tmp_vendor not in sys.path
        assert vc._VENDORED_MODULES == {}
    finally:
        _restore_fake_global_modules(saved)


def test_vendored_modules_are_used_and_cached(vc, tmp_vendor: str):
    _make_fake_vendor(tmp_vendor)
    vendored = {name: _fake_module(name, "vendored") for name in SWAPPED}
    vc._VENDORED_MODULES.clear()
    vc._VENDORED_MODULES.update(vendored)
    saved = _install_fake_global_modules()
    try:
        with vc.vendored_transformers():
            assert sys.modules["transformers"].__fake_marker__ == "vendored"
            assert sys.modules["huggingface_hub"].__fake_marker__ == "vendored"

        assert sys.modules["transformers"].__fake_marker__ == "global"
        # the vendored objects stay alive in the cache: identities must be
        # stable across node runs, or the cached model bundle would fail its
        # isinstance() checks against a rebuilt module family
        assert vc._VENDORED_MODULES["transformers"] is vendored["transformers"]

        with vc.vendored_transformers():
            assert sys.modules["transformers"] is vendored["transformers"]
        assert sys.modules["transformers"].__fake_marker__ == "global"
    finally:
        _restore_fake_global_modules(saved)
        vc._VENDORED_MODULES.clear()


def test_remote_code_modules_are_not_touched(vc, tmp_vendor: str):
    _make_fake_vendor(tmp_vendor)
    # remote code lives under `transformers_modules.*` (underscore!) -- it must
    # survive the swap, it is model code, not one of the swapped families
    saved = _install_fake_global_modules()
    remote = _fake_module("transformers_modules.Org.Model.processing", "remote")
    sys.modules[remote.__name__] = remote
    try:
        with vc.vendored_transformers():
            assert sys.modules.get(remote.__name__) is remote
            assert "transformers" not in sys.modules
        assert sys.modules.get(remote.__name__) is remote
    finally:
        sys.modules.pop(remote.__name__, None)
        _restore_fake_global_modules(saved)


def test_json_patch_scoped_to_the_context(vc, tmp_vendor: str):
    _make_fake_vendor(tmp_vendor)
    original_default = json.JSONEncoder.default
    saved = _install_fake_global_modules()
    try:
        outside_raises = False
        try:
            json.dumps({"c": _ConfigLike()})
        except TypeError:
            outside_raises = True  # the stdlib encoder, unpatched: config objects fail
        assert outside_raises
        with vc.vendored_transformers():
            inside = json.dumps({"c": _ConfigLike()})
            assert json.loads(inside)["c"] == {"model_type": "fake"}
        assert json.JSONEncoder.default is original_default
        with_raises = False
        try:
            json.dumps({"c": _ConfigLike()})
        except TypeError:
            with_raises = True
        assert with_raises
    finally:
        json.JSONEncoder.default = original_default
        _restore_fake_global_modules(saved)


def test_nested_contexts(vc, tmp_vendor: str):
    _make_fake_vendor(tmp_vendor)
    vendored = {name: _fake_module(name, "vendored") for name in SWAPPED}
    vc._VENDORED_MODULES.clear()
    vc._VENDORED_MODULES.update(vendored)
    saved = _install_fake_global_modules()
    try:
        with vc.vendored_transformers():
            with vc.vendored_transformers():
                assert sys.modules["transformers"] is vendored["transformers"]
                assert sys.path.count(tmp_vendor) >= 1
            assert sys.modules["transformers"] is vendored["transformers"]
            assert sys.path[0] == tmp_vendor  # outer context still active
        assert sys.modules["transformers"].__fake_marker__ == "global"
        assert tmp_vendor not in sys.path
    finally:
        _restore_fake_global_modules(saved)
        vc._VENDORED_MODULES.clear()


def test_pack_module_keys_matching(vc, tmp_vendor=None):
    keys = ["transformers", "transformers.models.qwen3.modeling_qwen3",
            "huggingface_hub", "hf_xet.internal",
            "transformers_modules.Org.Model.modeling",  # remote code: NOT swapped
            "tokenizers",  # Rust ext: NOT swapped
            "torch"]  # shared: NOT swapped
    for key in keys:
        sys.modules[key] = _fake_module(key, "x")
    try:
        matched = set(vc._pack_module_keys(vc.VENDOR_PACKAGES))
        assert matched == {
            "transformers", "transformers.models.qwen3.modeling_qwen3",
            "huggingface_hub", "hf_xet.internal",
        }
    finally:
        for key in keys:
            sys.modules.pop(key, None)


def test_nodes_py_binds_the_swap(vc, tmp_vendor=None):
    """nodes.py must end up with a working vendored_transformers even by PATH.

    The self-checks (and only they) load nodes.py as a bare module, where the
    relative `from .vendor_context import ...` cannot work; its fallback loads
    the sibling file directly. Inside ComfyUI the relative import wins -- but
    this check cannot see that without ComfyUI, and the fallback path is the
    fragile one.
    """
    spec = importlib.util.spec_from_file_location("moss_nodes_vendorcheck", ROOT / "nodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # the by-PATH fallback execs the sibling under its real name and binds the
    # context manager from there
    assert mod.vendored_transformers.__module__ == "vendor_context"
    assert callable(mod.vendored_transformers)

    # exactly the six methods that touch transformers/model bundles are wrapped
    wrapped = {
        "MOSSLoadModel.load", "MOSSSpeak.generate", "MOSSVoiceClone.generate",
        "MOSSVoiceContinue.generate", "MOSSEncodeTokens.encode", "MOSSDecodeTokens.decode",
    }
    seen = set()
    for class_name in ("MOSSLoadModel", "MOSSSpeak", "MOSSVoiceClone", "MOSSVoiceContinue",
                       "MOSSEncodeTokens", "MOSSDecodeTokens", "MOSSConcatTokens",
                       "MOSSSaveTokens", "MOSSLoadTokens", "MOSSEstimateTokens"):
        cls = getattr(mod, class_name)
        for attr, value in vars(cls).items():
            if callable(value) and getattr(value, "__wrapped__", None) is not None:
                seen.add(f"{class_name}.{attr}")
    assert seen == wrapped, f"wrapped set mismatch: extra={seen - wrapped} missing={wrapped - seen}"

    # ... and the node table itself is untouched by the decorator layer
    assert len(mod.NODE_CLASS_MAPPINGS) == 10


def _make_strict_processor_class():
    """A stand-in for the 5.x ProcessorMixin.__init__ (processing_utils.py).

    Mirrors the strict semantics that broke 4.x-era remote code: chat_template
    and audio_tokenizer are popped first, any other kwarg that is not a
    declared attribute raises ``TypeError: Unexpected keyword argument X.``,
    and the remaining kwargs must exactly match the declared attributes.
    """

    class StrictProcessor:
        def __init__(self, *args, **kwargs):
            setattr(self, "chat_template", kwargs.pop("chat_template", None))
            audio_tokenizer = kwargs.pop("audio_tokenizer", None)
            if audio_tokenizer is not None:
                self.audio_tokenizer = audio_tokenizer
            for key in kwargs:
                if key not in self.get_attributes():
                    raise TypeError(f"Unexpected keyword argument {key}.")
            for arg, attribute_name in zip(args, self.get_attributes()):
                if attribute_name in kwargs:
                    raise TypeError(f"Got multiple values for argument {attribute_name}.")
                kwargs[attribute_name] = arg
            if len(kwargs) != len(self.get_attributes()):
                raise ValueError("This processor requires the wrong number of arguments")
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def get_attributes(cls):
            return ["tokenizer"]

    return StrictProcessor


def test_tolerant_processor_init_covers_remote_code(vc, tmp_vendor=None):
    """The user-log failure, replayed: 5.x injects revision=, remote code
    (written for 4.x) forwards it into the strict __init__ -> TypeError.

    The patch must restore the 4.x behavior (unknown kwargs become plain
    attributes) WITHOUT touching what the original init consumes itself.
    """
    cls = _make_strict_processor_class()
    # sanity: unpatched, this is exactly the user's traceback
    try:
        cls(tokenizer="TOK", audio_tokenizer="CODEC", revision="be7766a6")
        raise AssertionError("the strict init should have raised")
    except TypeError as e:
        assert "Unexpected keyword argument revision" in str(e)

    cls.__init__ = vc._make_tolerant_processor_init(cls.__init__)

    # the patched one must construct the very same call
    p = cls(tokenizer="TOK", audio_tokenizer="CODEC", revision="be7766a6")
    assert p.tokenizer == "TOK"
    assert p.audio_tokenizer == "CODEC"
    assert p.revision == "be7766a6"  # 4.x behavior: kept as a plain attribute

    # several injected kwargs at once, and chat_template must survive the trip
    # through the wrapper (it belongs to the original init, not to "extras")
    p2 = cls(tokenizer="TOK", chat_template="CHATTY", revision="r1", token="t", cache_dir="/c")
    assert p2.tokenizer == "TOK"
    assert p2.chat_template == "CHATTY"
    assert p2.revision == "r1" and p2.token == "t" and p2.cache_dir == "/c"

    # positional args keep flowing to the zip(args, attributes) path
    p3 = cls("TOK")
    assert p3.tokenizer == "TOK"
    assert p3.chat_template is None


def test_tolerant_processor_init_clean_call_is_passthrough(vc, tmp_vendor=None):
    """A call with no unexpected kwargs must reach the original untouched."""
    calls = []

    def original_init(self, **kwargs):
        calls.append(dict(kwargs))
        self.from_original = True

    cls = type("Clean", (), {"get_attributes": classmethod(lambda cls: ["tokenizer"])})
    cls.__init__ = vc._make_tolerant_processor_init(original_init)

    c = cls(tokenizer="TOK")
    assert calls == [{"tokenizer": "TOK"}]
    assert c.from_original is True
    assert not hasattr(c, "revision")
    assert cls.__init__._moss_tts_tolerant_kwargs is True


def test_tolerant_processor_init_survives_broken_attributes(vc, tmp_vendor=None):
    """get_attributes() blowing up must not crash on our account: degrade to
    pass-through, let the original init raise its own (better) error."""
    calls = []

    def original_init(self, **kwargs):
        calls.append(dict(kwargs))

    def broken(cls):
        raise RuntimeError("boom")

    cls = type("Broken", (), {"get_attributes": classmethod(broken)})
    cls.__init__ = vc._make_tolerant_processor_init(original_init)

    c = cls(tokenizer="TOK", revision="r1")
    # everything outside chat_template/audio_tokenizer was stripped from the
    # original's kwargs but kept as attributes -- 4.x style, no TypeError
    assert calls == [{}]
    assert c.revision == "r1"


def test_native_packages_swap_only_when_vendored(vc, tmp_vendor: str):
    """tokenizers/safetensors enter the swap ONLY when vendor/ has a copy.

    Without a vendored copy the shared native module must be untouched (there
    is nothing to swap in, and the shared tokenizers belongs to the rest of
    ComfyUI). With one, it takes part in the same keep-alive swap.
    """
    saved = _install_fake_global_modules()
    saved["tokenizers"] = sys.modules.get("tokenizers")
    saved["safetensors"] = sys.modules.get("safetensors")
    sys.modules["tokenizers"] = _fake_module("tokenizers", "global")
    sys.modules["safetensors"] = _fake_module("safetensors", "global")

    def cleanup():
        for name, module in (("tokenizers", saved["tokenizers"]), ("safetensors", saved["safetensors"])):
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        _restore_fake_global_modules(saved)
        vc._VENDORED_MODULES.clear()

    try:
        # vendor/ WITHOUT native copies: natives stay shared inside the context
        _make_fake_vendor(tmp_vendor)
        with vc.vendored_transformers():
            assert sys.modules["tokenizers"].__fake_marker__ == "global"
            assert sys.modules["safetensors"].__fake_marker__ == "global"
        assert vc._VENDORED_MODULES.get("tokenizers") is None

        # vendor/ WITH a (fake) vendored copy of both: they join the swap
        os.makedirs(os.path.join(tmp_vendor, "tokenizers"), exist_ok=True)
        os.makedirs(os.path.join(tmp_vendor, "safetensors"), exist_ok=True)
        with vc.vendored_transformers():
            assert "tokenizers" not in sys.modules  # global out, nothing cached to re-insert yet
            assert "safetensors" not in sys.modules
        assert sys.modules["tokenizers"].__fake_marker__ == "global"
        assert sys.modules["safetensors"].__fake_marker__ == "global"
    finally:
        cleanup()


def test_ready_marker_round_trip(vc, tmp_vendor: str):
    """The .ready marker round-trips, and schema 1 (old vendor/) reads as
    NOT complete -- that is what migrates an old install without deleting."""
    vc.vendor_directory = lambda: tmp_vendor
    try:
        assert vc.read_ready_marker() is None  # nothing written yet

        vc.write_ready_marker(["tokenizers<0.24.0,>=0.23.1"])
        marker = vc.read_ready_marker()
        assert marker == {"schema": vc.VENDOR_READY_SCHEMA,
                          "vendored_gaps": ["tokenizers<0.24.0,>=0.23.1"]}

        # a stale/corrupt marker is not an error, just "not complete"
        with open(vc.ready_marker_path(), "w", encoding="utf-8") as broken:
            broken.write("{not json")
        assert vc.read_ready_marker() is None

        # schema 1 semantics: an old bare marker (or old schema) never counts
        # as a complete setup against the current schema
        assert marker.get("schema", 0) >= vc.VENDOR_READY_SCHEMA
        assert ({"schema": 1}.get("schema", 0) >= vc.VENDOR_READY_SCHEMA) is False
    finally:
        pass


def test_core_dependency_gaps_detection(vc, tmp_vendor: str):
    """Gap detection reads the VENDORED transformers' own Requires-Dist and
    compares with the SHARED environment's versions -- exactly the 0.22.2
    tokenizers vs >=0.23.1,<0.24.0 collision from the user's report."""
    dist_info = os.path.join(tmp_vendor, "transformers-5.20.2.dist-info")
    os.makedirs(dist_info, exist_ok=True)
    with open(os.path.join(dist_info, "METADATA"), "w", encoding="utf-8") as metadata:
        metadata.write("Metadata-Version: 2.2\nName: transformers\nVersion: 5.20.2\n")
        metadata.write("Requires-Dist: tokenizers<0.24.0,>=0.23.1\n")
        metadata.write("Requires-Dist: safetensors>=0.8.0\n")
        metadata.write('Requires-Dist: numpy; extra == "dev"\n')  # must be ignored
        metadata.write("Requires-Dist: httpx2<3,>=2.0.0\n")       # not a native: ignored

    def versions(table):
        return lambda name: table.get(name)

    # the user's shared environment: tokenizers 0.22.2, safetensors 0.5.3
    gaps = vc.core_dependency_gaps(
        vendor_dir=tmp_vendor,
        installed_version=versions({"tokenizers": "0.22.2", "safetensors": "0.5.3"}),
    )
    assert gaps == ["tokenizers<0.24.0,>=0.23.1", "safetensors>=0.8.0"], gaps

    # a shared environment that is new enough: no gaps, nothing vendored
    assert vc.core_dependency_gaps(
        vendor_dir=tmp_vendor,
        installed_version=versions({"tokenizers": "0.23.3", "safetensors": "0.8.0"}),
    ) == []

    # a package missing from the shared env entirely: still a gap
    assert vc.core_dependency_gaps(
        vendor_dir=tmp_vendor,
        installed_version=versions({}),
    ) == ["tokenizers<0.24.0,>=0.23.1", "safetensors>=0.8.0"]

    # unreadable METADATA: fall back to the verified requirement constants
    gaps_fallback = vc.core_dependency_gaps(
        vendor_dir=os.path.join(tmp_vendor, "does-not-exist"),
        installed_version=versions({"tokenizers": "0.22.2", "safetensors": "0.5.3"}),
    )
    assert gaps_fallback == [
        f"tokenizers{vc.FALLBACK_NATIVE_REQUIREMENTS['tokenizers']}",
        f"safetensors{vc.FALLBACK_NATIVE_REQUIREMENTS['safetensors']}",
    ]


def main() -> int:
    vc = _import_vendor_context()

    checks = [test_fallback_without_vendor_is_a_noop, test_swap_and_restore,
              test_vendored_modules_are_used_and_cached,
              test_remote_code_modules_are_not_touched, test_json_patch_scoped_to_the_context,
              test_nested_contexts, test_pack_module_keys_matching, test_nodes_py_binds_the_swap,
              test_tolerant_processor_init_covers_remote_code,
              test_tolerant_processor_init_clean_call_is_passthrough,
              test_tolerant_processor_init_survives_broken_attributes,
              test_native_packages_swap_only_when_vendored,
              test_ready_marker_round_trip,
              test_core_dependency_gaps_detection]

    with tempfile.TemporaryDirectory() as tmp_vendor:
        # point the swap at an empty fake vendor/ -- the context only needs the
        # PATH, it never imports from there itself
        vc.vendor_directory = lambda: tmp_vendor

        for check in checks:
            check(vc, tmp_vendor)
            print(f"  ok: {check.__name__}")

    # vendor/ state helpers agree with reality -- checked on a FRESH module
    # import, because `vc` above still has its vendor_directory monkeypatched
    # to the now-deleted temporary directory
    fresh = _import_vendor_context()
    assert fresh.has_vendored_transformers() is os.path.exists(
        os.path.join(ROOT, "vendor", "transformers", "__init__.py"))
    print("  ok: vendor marker check")
    print("vendor_context self-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
