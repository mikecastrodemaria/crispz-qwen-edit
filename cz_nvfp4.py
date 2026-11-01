"""Qwen-Image 2.1 Turbo in NVFP4, run natively (Blackwell tensor cores, through comfy-kitchen).

The existing dequant loader turns FP8/INT8 into bf16 and pays the full memory footprint.
NVFP4 is worth having only when it is NOT dequantized: measured on an RTX 5090 at 1024x1024,
one transformer pass takes 0.28 s and 6.1 GB natively against 0.46 s and 14.0 GB in bf16.
So the transformer keeps its packed weights here. The text encoder is dequantized to bf16:
public ComfyUI runs it in full precision too.

Needs its own environment (requirements-nvfp4.txt): Python >= 3.11, torch cu130 or later,
transformers 5, a recent diffusers and comfy-kitchen. Every heavy import is lazy, so this
module imports fine in the regular venv and runtime_problem() says what is missing.
"""
import types

DEFAULT_BASE_REPO = "Qwen/Qwen-Image-2.1-Turbo"
DEFAULT_ENCODER_REPO = "BennyDaBall/Qwen-Image-2.1-Turbo-NVFP4"
DEFAULT_ENCODER_FILE = "text_encoders/qwen3vl_8b_turbo_nvfp4.safetensors"

LAYOUT = "TensorCoreNVFP4Layout"
_QUANT_SUFFIXES = (".weight_scale", ".weight_scale_2", ".comfy_quant")
_FUSED_GATE = "img_mlp.gate_up.weight"
_MIN_CUDA = (13,)
_MIN_SM_MAJOR = 10
_PROMPT_MEMO_MAX = 8


def header_is_nvfp4(hdr):
    """True when a safetensors header holds NVFP4 weights: packed U8 X.weight beside the
    per-tensor X.weight_scale_2 that only this format has."""
    for k, v in (hdr or {}).items():
        if k.endswith(".weight_scale_2") and isinstance(v, dict):
            w = hdr.get(k[:-len("_scale_2")])
            if isinstance(w, dict) and str(w.get("dtype", "")).upper() == "U8":
                return True
    return False


def runtime_problem(check_device=True):
    """What prevents the native path in this environment (an actionable str), or None."""
    hint = "install requirements-nvfp4.txt in a dedicated venv (see README)"
    try:
        import torch
        import diffusers
        import comfy_kitchen  # noqa: F401
    except ImportError as e:
        return f"NVFP4 needs {e.name or e}: {hint}"
    if not hasattr(diffusers, "QwenImage21Pipeline"):
        return f"this diffusers ({diffusers.__version__}) has no QwenImage21Pipeline: {hint}"
    cuda = tuple(int(x) for x in str(torch.version.cuda or "0").split("."))
    if cuda < _MIN_CUDA:
        return f"NVFP4 needs torch built for CUDA 13 or later, found {torch.version.cuda}: {hint}"
    if check_device:
        if not torch.cuda.is_available():
            return "NVFP4 runs natively on an NVIDIA GPU only, and none is available"
        major = torch.cuda.get_device_properties(0).major
        if major < _MIN_SM_MAJOR:
            return (f"NVFP4 needs a Blackwell card (compute capability 10+), "
                    f"this one is {major}.x")
    return None


def base_keys(keys):
    """The checkpoint's real tensors, without its quantization side keys."""
    return {k for k in keys if not k.endswith(_QUANT_SUFFIXES)}


def expected_keys(keys):
    """The diffusers keys a checkpoint provides: ComfyUI fuses the SwiGLU gate and up
    projections into one matrix (gate rows first), diffusers declares two."""
    out = set()
    for k in base_keys(keys):
        if k.endswith(_FUSED_GATE):
            out |= {k.replace("gate_up", "gate_layer"), k.replace("gate_up", "proj")}
        else:
            out.add(k)
    return out


def encoder_key(k):
    """Text encoder key, ComfyUI layout -> Qwen3VLForConditionalGeneration."""
    for old in ("model.layers.", "model.embed_tokens", "model.norm"):
        if k.startswith(old):
            return "model.language_model." + k[len("model."):]
    return k


def _layer_class():
    import torch.nn as nn
    import torch.nn.functional as F
    from comfy_kitchen.tensor import QuantizedTensor

    class NVFP4Linear(nn.Module):
        """A linear layer on packed weights: the input is quantized too, the product runs
        on the tensor cores. Same inference path as ComfyUI's mixed_precision_ops.Linear."""

        def __init__(self, weight, bias=None):
            super().__init__()
            self.weight = weight
            self.bias = bias

        def forward(self, x):
            q = QuantizedTensor.from_float(x.flatten(0, -2), LAYOUT, scale=None)
            out = F.linear(q, self.weight, self.bias)
            return out.unflatten(0, x.shape[:-1]) if x.ndim > 2 else out

    return NVFP4Linear


def packed_tensor(f, key, dtype, device, shape=None):
    """QuantizedTensor for the X.weight `key` of an open safetensors file. A packed row holds
    two values per byte, hence the default shape."""
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout as L
    w = f.get_tensor(key).to(device)
    params = L.Params(scale=f.get_tensor(key + "_scale_2").to(device),
                      block_scale=f.get_tensor(key + "_scale").to(device),
                      orig_dtype=dtype,
                      orig_shape=tuple(shape) if shape else (w.shape[0], w.shape[1] * 2))
    return QuantizedTensor(w, LAYOUT, params)


def _fused_forward(self, x):
    gate, up = self.gate_up(x).chunk(2, dim=-1)
    return self.out(self.activation_fn(gate) * up)


def load_transformer(path, base_repo, dtype, device):
    """A QwenImage21Transformer2DModel whose quantized layers stay packed, on `device`."""
    import torch.nn as nn
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel as Model
    from safetensors import safe_open

    cfg = Model.load_config(base_repo, subfolder="transformer")
    with init_empty_weights():
        model = Model.from_config(cfg)
    want = set(model.state_dict())
    NVFP4Linear = _layer_class()
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        have = expected_keys(keys)
        if want != have:
            raise RuntimeError(
                f"{path}: not a Qwen-Image 2.1 transformer ({len(want - have)} missing keys, "
                f"e.g. {sorted(want - have)[:2]}; {len(have - want)} unknown, "
                f"e.g. {sorted(have - want)[:2]})")
        real = sorted(base_keys(keys))
        for k in real:
            if k + "_scale_2" not in keys:
                continue
            mod = k.rpartition(".")[0]
            parent_name, _, name = mod.rpartition(".")
            parent = model.get_submodule(parent_name)
            bias = None
            if mod + ".bias" in keys:
                bias = nn.Parameter(f.get_tensor(mod + ".bias").to(device, dtype),
                                    requires_grad=False)
            layer = NVFP4Linear(packed_tensor(f, k, dtype, device), bias)
            if name == "gate_up":
                # Keep the fused matrix: a packed tensor cannot be split without
                # dequantizing it, so the feed-forward chunks the OUTPUT instead.
                del parent.gate_layer, parent.proj
                parent.gate_up = layer
                parent.forward = types.MethodType(_fused_forward, parent)
            else:
                setattr(parent, name, layer)
        for k in real:
            mod, _, name = k.rpartition(".")
            owner = model.get_submodule(mod)
            if isinstance(owner, NVFP4Linear):
                continue
            t = f.get_tensor(k).to(device)
            t = t.to(dtype) if t.is_floating_point() else t
            if name in owner._parameters:
                setattr(owner, name, nn.Parameter(t, requires_grad=False))
            else:
                owner.register_buffer(name, t)
    return model.eval()


def load_text_encoder(path, base_repo, dtype, device):
    """The Qwen3-VL encoder dequantized to bf16, on `device`."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration as Encoder

    cfg = AutoConfig.from_pretrained(base_repo, subfolder="text_encoder")
    with init_empty_weights():
        model = Encoder(cfg)
    target = model.state_dict()
    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for k in base_keys(keys):
            name = encoder_key(k)
            if name not in target:
                raise RuntimeError(f"{path}: unknown text encoder key {k}")
            if k + "_scale_2" in keys:
                t = packed_tensor(f, k, dtype, device, target[name].shape).dequantize()
            else:
                t = f.get_tensor(k).to(device)
            sd[name] = t.to(dtype) if t.is_floating_point() else t
    missing = model.load_state_dict(sd, strict=False, assign=True).missing_keys
    model.tie_weights()
    still = [n for n, p in model.named_parameters() if p.is_meta]
    if still:
        raise RuntimeError(f"{path}: text encoder incomplete, e.g. {still[:2]} "
                           f"(missing: {missing[:2]})")
    return model.eval()


def encoder_on_demand(pipe, device):
    """Keeps the text encoder in RAM and brings it to `device` for each encoding only: resident,
    its 17 GB leave no room for an edit at 1024x1024 on a 32 GB card.
    That round trip costs about 5 s, so text-only prompts are remembered in RAM (a few MB, for
    the life of this pipe). An edit's encoding reads its image and is never remembered."""
    import torch
    encoder = pipe.text_encoder
    encode = pipe.encode_prompt
    encoder.to("cpu")
    memo = {}

    def on_device(values):
        return tuple(v.to(device) if hasattr(v, "to") else v for v in values)

    def encode_on_device(*args, **kwargs):
        prompt = kwargs.get("prompt", args[0] if args else None)
        key = None
        if (isinstance(prompt, str) and len(args) <= 1 and kwargs.get("image") is None
                and kwargs.get("prompt_embeds") is None):
            key = (prompt, kwargs.get("num_images_per_prompt", 1))
            if key in memo:
                return on_device(memo[key])
        encoder.to(device)
        try:
            out = encode(*args, **kwargs)
        finally:
            encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if key is not None:
            if len(memo) >= _PROMPT_MEMO_MAX:
                memo.pop(next(iter(memo)))
            memo[key] = tuple(v.to("cpu") if hasattr(v, "to") else v for v in out)
        return out

    pipe.encode_prompt = encode_on_device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_encoder(source=""):
    """A local encoder file when one is configured, otherwise the Hub's (downloaded once)."""
    if source:
        return source
    from huggingface_hub import hf_hub_download
    return hf_hub_download(DEFAULT_ENCODER_REPO, DEFAULT_ENCODER_FILE)
