"""Unit tests for the native NVFP4 path (cz_nvfp4 + its branches in cz_pipeline).

No model is loaded. Only the layer test touches the GPU, and it is skipped without one
(CUDA_VISIBLE_DEVICES=-1 hides the card; an empty value does not survive on Windows) or outside the NVFP4 venv.

Run:  .venv/Scripts/python tests/test_nvfp4.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_nvfp4 as N  # noqa: E402
import cz_pipeline as P  # noqa: E402


def test_header_detection():
    nvfp4 = {"a.weight": {"dtype": "U8"}, "a.weight_scale": {"dtype": "F8_E4M3"},
             "a.weight_scale_2": {"dtype": "F32"}, "__metadata__": {"format": "pt"}}
    fp8 = {"a.weight": {"dtype": "F8_E4M3"}, "a.weight_scale": {"dtype": "F32"}}
    assert N.header_is_nvfp4(nvfp4)
    assert not N.header_is_nvfp4(fp8)
    assert not N.header_is_nvfp4({})


def test_expected_keys_split_the_fused_gate():
    keys = {"transformer_blocks.0.img_mlp.gate_up.weight",
            "transformer_blocks.0.img_mlp.gate_up.weight_scale",
            "transformer_blocks.0.img_mlp.gate_up.weight_scale_2",
            "transformer_blocks.0.img_mlp.gate_up.comfy_quant",
            "transformer_blocks.0.img_mlp.out.weight"}
    assert N.expected_keys(keys) == {"transformer_blocks.0.img_mlp.gate_layer.weight",
                                     "transformer_blocks.0.img_mlp.proj.weight",
                                     "transformer_blocks.0.img_mlp.out.weight"}


def test_encoder_keys():
    assert N.encoder_key("model.layers.3.mlp.up_proj.weight") == "model.language_model.layers.3.mlp.up_proj.weight"
    assert N.encoder_key("model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert N.encoder_key("model.norm.weight") == "model.language_model.norm.weight"
    assert N.encoder_key("model.visual.blocks.0.norm1.weight") == "model.visual.blocks.0.norm1.weight"
    assert N.encoder_key("lm_head.weight") == "lm_head.weight"


class FakePipe:
    _cz_nvfp4_path = "x.safetensors"

    def __init__(self):
        self.kw = None

    def __call__(self, **kw):
        self.kw = kw
        return "out"


def test_call_forces_the_turbo_settings():
    pipe = FakePipe()
    out = P._qwen_call(pipe, prompt="a cat", true_cfg_scale=4.0, negative_prompt="ugly",
                       callback_on_step_end=print,
                       callback_on_step_end_tensor_inputs=["latents"])
    assert out == "out"
    assert pipe.kw == {"prompt": "a cat", "true_cfg_scale": 1.0, "negative_prompt": None}


def test_edit_loras_are_refused_not_dropped():
    saved = P.EDIT_LORAS, P.EDIT_LORAS_ENABLED, P.EDIT_SPEED
    P.EDIT_LORAS, P.EDIT_LORAS_ENABLED, P.EDIT_SPEED = [("a.safetensors", 1.0)], True, None
    try:
        P._apply_edit_loras(FakePipe())
    except RuntimeError as e:
        assert "NVFP4" in str(e)
    else:
        raise AssertionError("an edit without its LoRA would be a false result")
    finally:
        P.EDIT_LORAS, P.EDIT_LORAS_ENABLED, P.EDIT_SPEED = saved


def test_encoder_only_visits_the_device_while_encoding():
    moves = []

    class Encoder:
        def to(self, device):
            moves.append(device)

    class Pipe:
        text_encoder = Encoder()

        def encode_prompt(self, prompt, image=None):
            moves.append(f"encode {prompt}")
            if prompt == "boom":
                raise ValueError(prompt)
            return (prompt, None)

    pipe = Pipe()
    N.encoder_on_demand(pipe, "gpu")
    assert moves == ["cpu"]
    assert pipe.encode_prompt("a") == ("a", None)
    assert moves == ["cpu", "gpu", "encode a", "cpu"]
    moves.clear()
    assert pipe.encode_prompt(prompt="a") == ("a", None)
    assert moves == []                               # remembered: the encoder stays in RAM
    pipe.encode_prompt("a", image="photo")
    pipe.encode_prompt("a", image="photo")
    assert moves.count("encode a") == 2              # an edit reads its image: never remembered
    moves.clear()
    for i in range(N._PROMPT_MEMO_MAX + 1):
        pipe.encode_prompt(f"p{i}")
    moves.clear()
    pipe.encode_prompt("p0")
    assert moves == ["gpu", "encode p0", "cpu"]      # bounded: the oldest one left
    moves.clear()
    try:
        pipe.encode_prompt("boom")
    except ValueError:
        pass
    assert moves == ["gpu", "encode boom", "cpu"]    # a failure must not strand it in VRAM


def test_regular_venv_gets_an_actionable_reason():
    problem = N.runtime_problem(check_device=False)
    assert problem is None or "requirements-nvfp4" in problem


def test_native_layer_matches_the_dequantized_one():
    if N.runtime_problem():
        print("  skipped (needs the NVFP4 venv and a Blackwell GPU: torch's CPU refuses this product)")
        return
    import torch
    import torch.nn.functional as F
    from comfy_kitchen.tensor import QuantizedTensor
    torch.manual_seed(0)
    w = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    weight = QuantizedTensor.from_float(w, N.LAYOUT, scale=None)
    x = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)
    out = N._layer_class()(weight).forward(x)
    ref = F.linear(x, weight.dequantize())
    assert out.shape == (2, 3, 32)
    assert F.cosine_similarity(out.flatten(), ref.flatten(), dim=0) > 0.95


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
