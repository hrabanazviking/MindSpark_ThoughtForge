"""TF-070 — ONNX export path truth.

Pins: the path format + live checker; the duties as
data (embedding models export; language models never
do; quantization is optimum-only); the exportable set
== TF-053's registry exactly; this machine's capability
measured live by find_spec (never filed); the profile
rows (phone_low consistent; pi_zero's flag without an
assignment = D-ONNX-2, filed for TF-086); and the
exportable set exercised end-to-end through STUB
optimum / torch / transformers / onnxruntime modules —
the real exporter logic (priority, D-ONNX-1's fixes:
config-probed dimension, quantized reported only when
a quantized file really landed, a torch-path quantize
request announced and reported False) and the real
embedder logic (plain file preferred, quantized the
recorded fallback, mean pooling over the attention
mask, L2 normalisation, empty input). No real export
runs: this VM carries none of the packages, and that
absence is itself pinned where it holds.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from thoughtforge.etl.embedding_discipline import EMBEDDING_MODELS
from thoughtforge.inference.onnx_path import (
    EXPORT_PACKAGES,
    PATH_DUTIES,
    PATH_FORMAT,
    PROFILE_ONNX_GAPS,
    capability,
    exportable_models,
    onnx_path_as_data,
    onnx_path_problems,
    package_status,
    profile_onnx_rows,
)


# ── Stub-module fixtures ────────────────────────────────────────────────────

@contextmanager
def stub_modules(mapping):
    """Install stub modules (None = import must fail),
    restoring every touched sys.modules entry after."""
    saved = {name: sys.modules.get(name, "___absent___") for name in mapping}
    for name, module in mapping.items():
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous == "___absent___":
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def fake_transformers(dim=384, vocab=30522, config_raises=False):
    module = types.ModuleType("transformers")

    class FakeTokenizer:
        vocab_size = vocab

        def __call__(self, texts, **kwargs):
            rows = texts if isinstance(texts, list) else [texts]
            return {
                "input_ids": np.full((len(rows), 4), 7, dtype=np.int64),
                "attention_mask": np.array(
                    [[1, 1, 1, 0]] * len(rows), dtype=np.int64
                ),
            }

        def save_pretrained(self, directory):
            Path(directory, "tokenizer.json").write_text("{}")

    class _TokenizerFactory:
        @staticmethod
        def from_pretrained(name):
            return FakeTokenizer()

    class _Config:
        hidden_size = dim

    class _ConfigFactory:
        @staticmethod
        def from_pretrained(name):
            if config_raises:
                raise RuntimeError("no config here")
            return _Config()

    class _Model:
        def __init__(self):
            self.config = _Config()

        def eval(self):
            return self

    class _ModelFactory:
        @staticmethod
        def from_pretrained(name):
            return _Model()

    module.AutoTokenizer = _TokenizerFactory
    module.AutoConfig = _ConfigFactory
    module.AutoModel = _ModelFactory
    return module


def fake_optimum(quantizer="writes"):
    """quantizer: 'writes' | 'writes_nothing' | 'absent'."""
    optimum = _package("optimum")
    exporters = _package("optimum.exporters")
    exporters_onnx = types.ModuleType("optimum.exporters.onnx")

    def main_export(model_name_or_path, output, task, opset, optimize):
        assert task == "feature-extraction"
        Path(output, "model.onnx").write_bytes(b"ONNX-FIXTURE")

    exporters_onnx.main_export = main_export
    onnxruntime_mod = _package("optimum.onnxruntime")
    configuration = types.ModuleType("optimum.onnxruntime.configuration")

    class _QConfig:
        @staticmethod
        def avx512_vnni(is_static=False):
            return object()

    configuration.AutoQuantizationConfig = _QConfig
    if quantizer != "absent":
        class _Quantizer:
            @staticmethod
            def from_pretrained(directory):
                return _Quantizer()

            def quantize(self, save_dir, quantization_config):
                if quantizer == "writes":
                    Path(save_dir, "model_quantized.onnx").write_bytes(
                        b"ONNX-FIXTURE-Q"
                    )

        onnxruntime_mod.ORTQuantizer = _Quantizer
    onnxruntime_mod.configuration = configuration
    return {
        "optimum": optimum,
        "optimum.exporters": exporters,
        "optimum.exporters.onnx": exporters_onnx,
        "optimum.onnxruntime": onnxruntime_mod,
        "optimum.onnxruntime.configuration": configuration,
    }


def fake_torch():
    torch = _package("torch")
    onnx_mod = types.ModuleType("torch.onnx")

    import contextlib

    @contextlib.contextmanager
    def no_grad():
        yield

    def export(model, args, f, **kwargs):
        Path(f).write_bytes(b"ONNX-FIXTURE-TORCH")

    torch.no_grad = no_grad
    torch.onnx = onnx_mod
    onnx_mod.export = export
    return {"torch": torch, "torch.onnx": onnx_mod}


def fake_onnxruntime():
    ort = types.ModuleType("onnxruntime")

    class _Output:
        shape = [None, None, 2]

    class Session:
        def __init__(self, path, providers):
            self.path = path

        def get_outputs(self):
            return [_Output()]

        def run(self, _outputs, inputs):
            batch = inputs["input_ids"].shape[0]
            hidden = np.zeros((batch, 4, 2), dtype=np.float32)
            hidden[:, 0] = [1.0, 0.0]
            hidden[:, 1] = [0.0, 1.0]
            hidden[:, 2] = [1.0, 1.0]
            hidden[:, 3] = [100.0, 100.0]   # padding: must be ignored
            return [hidden]

    ort.InferenceSession = Session
    return ort


NO_OPTIMUM = {
    "optimum": None, "optimum.exporters": None,
    "optimum.exporters.onnx": None, "optimum.onnxruntime": None,
    "optimum.onnxruntime.configuration": None,
}


# ── Truth as data ───────────────────────────────────────────────────────────

def test_format_and_live_checker():
    assert PATH_FORMAT == "thoughtforge-onnx-path/1"
    assert onnx_path_problems() == []


def test_duties_as_data():
    assert PATH_DUTIES["export_embedding_model"]["does"] is True
    assert PATH_DUTIES["quantize_int8"]["does"] is True
    assert PATH_DUTIES["embed_from_export"]["does"] is True
    assert PATH_DUTIES["export_language_model"]["does"] is False
    assert PATH_DUTIES["quantize_on_torch_path"]["does"] is False
    assert PATH_DUTIES["run_without_optional_packages"]["does"] is False


def test_exportable_set_is_exactly_the_registry():
    models = exportable_models()
    assert set(models) == set(EMBEDDING_MODELS) == {"all-MiniLM-L6-v2"}
    assert models["all-MiniLM-L6-v2"]["dimension"] == 384
    assert models["all-MiniLM-L6-v2"]["hub_id"] == \
        "sentence-transformers/all-MiniLM-L6-v2"
    assert models["all-MiniLM-L6-v2"]["licence"] == "Apache-2.0"


def test_capability_is_measured_not_filed():
    status = package_status()
    cap = capability()
    assert cap["packages"] == status
    for leg, packages in EXPORT_PACKAGES.items():
        assert cap["legs"][leg] == all(status[p] for p in packages)
    assert cap["export_possible"] == (
        cap["legs"]["optimum_path"] or cap["legs"]["torch_path"]
    )


def test_profile_rows_and_the_filed_gap():
    rows = profile_onnx_rows()
    assert rows["phone_low"] == {
        "deployment_onnx_export": True,
        "embedding_runtime": "onnx",
        "verdict": "consistent",
    }
    # D-ONNX-2: pi_zero declares the flag but TF-053
    # assigns it no embedding runtime at all.
    assert rows["pi_zero"]["deployment_onnx_export"] is True
    assert rows["pi_zero"]["embedding_runtime"] == "none"
    assert rows["pi_zero"]["verdict"] == "flag_without_assignment"
    gaps = {pid for pid, r in rows.items()
            if r["verdict"] not in ("consistent", "consistent_none")}
    assert gaps == set(PROFILE_ONNX_GAPS) == {"pi_zero"}
    for pid in ("desktop_cpu", "desktop_gpu", "pi_5", "server_gpu"):
        assert rows[pid]["deployment_onnx_export"] is False


def test_as_data_shape():
    data = onnx_path_as_data()
    assert data["format"] == PATH_FORMAT
    assert data["profile_gaps"] == ["pi_zero"]
    assert set(data["exportable_models"]) == {"all-MiniLM-L6-v2"}


# ── Fixtures: the exportable set, exercised ─────────────────────────────────

def test_fixture_optimum_export_probes_dimension(tmp_path):
    from thoughtforge.inference.onnx_export import OnnxExporter

    stubs = fake_optimum()
    stubs["transformers"] = fake_transformers(dim=768)
    with stub_modules(stubs):
        result = OnnxExporter().export_embedder(
            "fixture/model-x", output_dir=tmp_path
        )
    assert result.exporter_used == "optimum"
    assert result.quantized is False
    assert result.onnx_path == tmp_path / "model.onnx"
    assert result.onnx_path.exists()
    # D-ONNX-1: the dimension is the config's 768,
    # not the old hardcoded 384.
    assert result.embedding_dim == 768
    assert result.vocab_size == 30522


def test_fixture_dimension_falls_back_to_registry(tmp_path):
    from thoughtforge.inference.onnx_export import OnnxExporter

    stubs = fake_optimum()
    stubs["transformers"] = fake_transformers(config_raises=True)
    with stub_modules(stubs):
        result = OnnxExporter().export_embedder(output_dir=tmp_path)
    assert result.embedding_dim == 384   # the registry's, for THIS model


def test_fixture_optimum_quantized_when_file_lands(tmp_path):
    from thoughtforge.inference.onnx_export import OnnxExporter

    stubs = fake_optimum(quantizer="writes")
    stubs["transformers"] = fake_transformers()
    with stub_modules(stubs):
        result = OnnxExporter().export_embedder(
            output_dir=tmp_path, quantize=True
        )
    assert result.quantized is True
    assert result.onnx_path == tmp_path / "model_quantized.onnx"


def test_fixture_quantize_request_never_lies(tmp_path):
    """D-ONNX-1: quantizer absent, or writing nothing —
    the result reports the plain model, quantized False."""
    from thoughtforge.inference.onnx_export import OnnxExporter

    for mode in ("absent", "writes_nothing"):
        stubs = fake_optimum(quantizer=mode)
        stubs["transformers"] = fake_transformers()
        with stub_modules(stubs):
            result = OnnxExporter().export_embedder(
                output_dir=tmp_path, quantize=True
            )
        assert result.quantized is False, mode
        assert result.onnx_path == tmp_path / "model.onnx", mode


def test_fixture_torch_fallback_cannot_quantize(tmp_path):
    from thoughtforge.inference.onnx_export import OnnxExporter

    stubs = dict(NO_OPTIMUM)
    stubs.update(fake_torch())
    stubs["transformers"] = fake_transformers(dim=512)
    with stub_modules(stubs):
        result = OnnxExporter().export_embedder(
            "fixture/model-y", output_dir=tmp_path, quantize=True
        )
    assert result.exporter_used == "torch"
    # The request was not honoured — and not claimed.
    assert result.quantized is False
    assert result.onnx_path == tmp_path / "model.onnx"
    assert result.embedding_dim == 512   # the model's own config


def test_fixture_embedder_prefers_plain_and_records_it(tmp_path):
    from thoughtforge.inference.onnx_export import ONNXEmbedder

    (tmp_path / "model.onnx").write_bytes(b"ONNX-FIXTURE")
    (tmp_path / "model_quantized.onnx").write_bytes(b"ONNX-FIXTURE-Q")
    stubs = {
        "onnxruntime": fake_onnxruntime(),
        "transformers": fake_transformers(),
    }
    with stub_modules(stubs):
        embedder = ONNXEmbedder(tmp_path)
    assert embedder.onnx_path == tmp_path / "model.onnx"
    assert embedder.quantized is False


def test_fixture_embedder_quantized_fallback_and_pooling(tmp_path):
    from thoughtforge.inference.onnx_export import ONNXEmbedder

    (tmp_path / "model_quantized.onnx").write_bytes(b"ONNX-FIXTURE-Q")
    stubs = {
        "onnxruntime": fake_onnxruntime(),
        "transformers": fake_transformers(),
    }
    with stub_modules(stubs):
        embedder = ONNXEmbedder(tmp_path)
        vectors = embedder.encode(["fehu", "uruz"])
        empty = embedder.encode([])
    assert embedder.onnx_path == tmp_path / "model_quantized.onnx"
    assert embedder.quantized is True
    # Mean over the 3 real positions ([1,0],[0,1],[1,1])
    # = [2/3, 2/3]; the padded position's [100,100] is
    # excluded by the mask; L2-normalised -> [~.707, ~.707].
    assert vectors.shape == (2, 2)
    assert np.allclose(vectors[0], [2 ** -0.5, 2 ** -0.5], atol=1e-6)
    assert empty.shape == (0, 2)


# ── This machine's truth, where it holds ────────────────────────────────────

def test_this_machine_exports_nothing_without_packages(tmp_path):
    if capability()["export_possible"]:
        pytest.skip("an exporter package is present on this machine")
    from thoughtforge.inference.onnx_export import OnnxExporter

    with pytest.raises(ImportError, match="ONNX export requires"):
        OnnxExporter().export_embedder(output_dir=tmp_path)


def test_this_machine_embeds_nothing_without_packages(tmp_path):
    if capability()["embed_possible"]:
        pytest.skip("the embedder packages are present on this machine")
    from thoughtforge.inference.onnx_export import ONNXEmbedder

    with pytest.raises(ImportError, match="onnxruntime is required"):
        ONNXEmbedder(tmp_path)
