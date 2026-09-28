"""crispz-qwen-edit - the Qwen-Image core (diffusers, BF16): loading the pipelines
(txt2img / img2img / inpaint) + instruction-based editing (the Omni/Edit tab), LoRAs /
checkpoints / transformer, generation and orchestration (generate / txt2img_run /
process_one / outpaint / inpaint) + the mutable runtime state.

Fork of crispz-studio (Z-Image). Mapping:
  - base txt2img             -> QwenImagePipeline
  - img2img (refine/upscale) -> QwenImageImg2ImgPipeline
  - inpaint / reframe        -> QwenImageInpaintPipeline
  - the Omni/Edit tab        -> QwenImageEditPlusPipeline (a SEPARATE model, multi-image,
                                default 'Qwen/Qwen-Image-Edit-2509') through generate_omni.
Every Qwen pipeline uses a REAL CFG (`true_cfg_scale`) + negative_prompt; so the UI's
"guidance" slider drives true_cfg_scale (see _cfg / _qwen_call), and the distilled
`guidance_scale` stays at 1.0. The module's public API stays identical to upstream's (the
same names, e.g. ZIMAGE_TRANSFORMER, generate_omni, SAMPLER_CHOICES) so that neither cz_ui
nor cz_cli breaks.

app reads the current state through cz_pipeline.NAME (BASE_REPO, ZIMAGE_TRANSFORMER, ...)
and sets cz_pipeline._PROGRESS / cz_pipeline._STOP from the UI handlers.
Depends only on cz_core / cz_esrgan / cz_imageio (never on app or gradio).

"""

import os
import gc
import sys
import time
import json
import threading

import numpy as np
import torch
from PIL import Image

import cz_core
from cz_core import (
    CONFIG, HERE, DEVICE, DTYPE,
    DEFAULT_TILE, DEFAULT_OVERLAP, DEFAULT_REFINE_TILE, DEFAULT_REFINE_OVERLAP,
    _prefs, _is_single_file, _log, _dbg,
)

# The base Qwen model (txt2img/img2img/inpaint). Overridable through env ZIMAGE_MODEL (compat)
# ou QWEN_MODEL, ou prefs. Repo public.
DEFAULT_BASE_REPO = (os.environ.get("QWEN_MODEL") or "Qwen/Qwen-Image")
# The instruction-based edit model (the Omni/Edit tab), loaded separately. 2509 = revision
# recente, multi-images. Surcharge via env ZIMAGE_OMNI_MODEL / QWEN_EDIT_MODEL ou config.
DEFAULT_OMNI_REPO = (os.environ.get("QWEN_EDIT_MODEL") or "Qwen/Qwen-Image-Edit-2509")
from cz_esrgan import load_esrgan, esrgan_upscale
from cz_imageio import _now_stamp
import cz_hw

# Speed: allow TF32 (matmul/cudnn) on the GPU. A free win on Ampere+ for the residual
# fp32 operations; the weights stay BF16. No effect outside CUDA.
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# The current Z-Image model. An HF repo / diffusers folder -> BASE_REPO. A single-file
# checkpoint (a Civitai .safetensors) passed as the "model" -> a transformer override (the
# VAE and the Qwen3 encoder still come from the base repo).
_zmodel = os.environ.get("ZIMAGE_MODEL") or _prefs.get("zimage_model") or DEFAULT_BASE_REPO
ZIMAGE_TRANSFORMER = os.environ.get("ZIMAGE_TRANSFORMER") or _prefs.get("zimage_transformer") or None
if _is_single_file(_zmodel):
    ZIMAGE_TRANSFORMER = _zmodel
    BASE_REPO = DEFAULT_BASE_REPO
else:
    BASE_REPO = _zmodel

# Replacement text encoder (Models > Checkpoints > Text encoder). Empty = the repo's,
# as before. Otherwise a FOLDER in transformers format (config.json + weights) or an HF repo
# ('owner/repo', 'owner/repo/subfolder') -- e.g. an "abliterated" Qwen2.5-VL-7B.
# It holds for BOTH pipes, base and edit. Only the encoder changes: tokenizer, processor,
# VAE and transformer stay the repo's.
CFG_TEXT_ENCODER_KEY = "text_encoder"


def _resolve_text_encoder(env, prefs, config):
    """The encoder at startup: env > preferences > config. A key PRESENT in the
    preferences wins even when empty: that is the "Default" choice made in the UI, and a
    config.txt value must not undo it on the next start (a "" used to pass for absent)."""
    v = str(env.get("QWEN_TEXT_ENCODER") or "").strip()
    if v:
        return v
    if CFG_TEXT_ENCODER_KEY in prefs:
        return str(prefs.get(CFG_TEXT_ENCODER_KEY) or "").strip()
    return str(config.get(CFG_TEXT_ENCODER_KEY) or "").strip()


TEXT_ENCODER = _resolve_text_encoder(os.environ, _prefs, CONFIG)
# The one REALLY loaded ('' = the repo's), per pipe: the base one (txt2img and its
# derived img2img / inpaint) and the edit pipe, which is loaded separately and can drop it
# on its own. Distinct from TEXT_ENCODER: an encoder that does not suit is dropped at load
# time, and the metadata says what ran, not what was asked for.
_TEXT_ENCODER_ACTIVE = ""
_TEXT_ENCODER_ACTIVE_EDIT = ""
TEXT_ENCODERS_DIR = str(os.environ.get("TEXT_ENCODERS_DIR") or _prefs.get("text_encoders_dir")
                        or CONFIG.get("text_encoders_dir") or "").strip()

# Z-Image model folders: single-file checkpoints to switch between + LoRAs to apply.
CHECKPOINTS_DIR = (os.environ.get("CHECKPOINTS_DIR") or _prefs.get("checkpoints_dir")
                   or CONFIG.get("checkpoints_dir") or os.path.join(HERE, "checkpoints"))
# Additional checkpoints folder (optional) -> merged with CHECKPOINTS_DIR into the same
# checkpoint list. Empty by default; configurable through UI / prefs / config / env.
CHECKPOINTS_EXTRA_DIR = (os.environ.get("CHECKPOINTS_EXTRA_DIR") or _prefs.get("checkpoints_extra_dir")
                         or CONFIG.get("checkpoints_extra_dir") or "").strip()
LORAS_DIR = (os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
             or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))


def _split_dirs(spec):
    """Folder list from a JSON list or from an 'a;b' string (os.pathsep or ';')."""
    if not spec:
        return []
    if isinstance(spec, str):
        parts = [p for chunk in spec.split(os.pathsep) for p in chunk.split(";")]
    else:
        parts = list(spec)
    out = []
    for p in parts:
        p = str(p or "").strip()
        if p and p not in out:
            out.append(p)
    return out


# EXTRA LoRA folders (e.g. the Civitai library shared with other tools): env
# LORAS_EXTRA_DIRS ('a;b') > preferences > config 'loras_extra_dirs'. Merged with
# LORAS_DIR into a single list; on a duplicate name, LORAS_DIR wins.
LORAS_EXTRA_DIRS = _split_dirs(os.environ["LORAS_EXTRA_DIRS"] if "LORAS_EXTRA_DIRS" in os.environ
                               else (_prefs.get("loras_extra_dirs")
                                     or CONFIG.get("loras_extra_dirs")))


def _lora_dirs():
    """LoRA folders to scan: the main one + the extras, deduplicated, in priority order."""
    dirs = [LORAS_DIR]
    for d in LORAS_EXTRA_DIRS:
        if d and d not in dirs:
            dirs.append(d)
    return dirs


def resolve_lora_path(name):
    """Path of a LoRA from a slot name: an absolute path as is, otherwise the relative
    name (subfolders included) looked up in LORAS_DIR then in the extras. Absent
    everywhere, the path inside LORAS_DIR (the caller reports 'not found')."""
    name = str(name or "")
    if os.path.isabs(name):
        return name
    for d in _lora_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(LORAS_DIR, name)


# Active LoRAs: a list of (path, weight). Several LoRAs can be combined (multi-slot).
LORAS = []
LORA_WEIGHT = float(CONFIG.get("default_lora_weight", 1.0))  # the slots' default weight


def _lora_weight_range():
    """Bounds of the LoRA weight sliders (config 'lora_weight_min'/'lora_weight_max').
    Default -2..2: NEGATIVE weights are valid and useful (they invert the LoRA's
    effect). Defensive: unreadable values or min >= max -> fall back to the default."""
    try:
        lo = float(CONFIG.get("lora_weight_min", -2.0))
        hi = float(CONFIG.get("lora_weight_max", 2.0))
    except (TypeError, ValueError):
        _log("lora_weight_min/max: not a number, using -2..2")
        return -2.0, 2.0
    if lo >= hi:
        _log(f"lora_weight_min ({lo}) >= lora_weight_max ({hi}), using -2..2")
        return -2.0, 2.0
    return lo, hi


LORA_WEIGHT_MIN, LORA_WEIGHT_MAX = _lora_weight_range()
# The default weight has to stay inside the bounds (or the slider would be born out of range).
LORA_WEIGHT = min(LORA_WEIGHT_MAX, max(LORA_WEIGHT_MIN, LORA_WEIGHT))
# LoRAs applied AT STARTUP (e.g. Lightning 8-step). config 'default_loras' = a list of
# names (inside LORAS_DIR) or of [name, weight] pairs. Resolved to (path, weight).
for _spec in (CONFIG.get("default_loras") or []):
    _nm, _w = (_spec if isinstance(_spec, (list, tuple)) and len(_spec) == 2
               else (_spec, LORA_WEIGHT))
    if _nm and _nm not in ("None", "none"):
        _p = resolve_lora_path(_nm)
        if os.path.isfile(_p):
            LORAS.append((_p, float(_w)))
# The Omni/Edit model (Qwen-Image-Edit, multi-image). The default = DEFAULT_OMNI_REPO
# so that the Edit tab works with no config. Tunable through config.txt
# (zimage_omni_model) or the UI.
OMNI_MODEL = (os.environ.get("ZIMAGE_OMNI_MODEL") or _prefs.get("zimage_omni_model")
              or CONFIG.get("zimage_omni_model") or DEFAULT_OMNI_REPO).strip()

# Process-wide caches. A "base" pipeline (txt2img ZImagePipeline) owns the
# components; img2img / inpaint derive from it through from_pipe -> shared weights, no
# duplicate VRAM. Cache key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE, LORAS).
_BASE_PIPE = None
_DERIVED = {}
_LOADED_KEY = None
# LoRAs actually applied on _BASE_PIPE (a list of (path, weight)). Used to hot-swap the
# LoRAs without reloading the model: if it diverges from LORAS, _apply_loras resyncs.
_APPLIED_LORAS = []
# EDIT LoRAs (the omni / Qwen-Image-Edit pipe): a set SEPARATE from the base one,
# because the edit transformer is another model (the cz_edit_loras presets target
# 2509/2511). Same (path, weight) format. EDIT_LORAS_ENABLED = the UI's "Edit LoRAs"
# checkbox: OFF -> the set is remembered but not applied (so with/without can be compared in
# one click).
EDIT_LORAS = []
EDIT_LORAS_ENABLED = bool(CONFIG.get("edit_loras_enabled", True))
_APPLIED_EDIT_LORAS = []
# FAST edit mode (the 'Edit speed' dropdown): None = off (steps/guidance from
# Settings), otherwise {"name", "steps", "guidance", "path"} - path = the Lightning LoRA
# to stack on the edit pipe (None for 'Auto': an already distilled model, Rapid-AIO or a
# Lightning merge, whose model_profiles profile sets steps/guidance).
EDIT_SPEED = None

# Step 2 (VRAM coexistence): CPU offload of the diffusion pass. none = everything in
# VRAM (the fastest). model = unloads per submodule (a good compromise). sequential = more
# aggressive, slower. This is NOT quantization: the weights stay BF16, they travel
# RAM <-> GPU. 'auto' (the default) = a free-VRAM test at load time (cz_hw): a model that
# overflows the VRAM does not crash, it spills into shared RAM (Windows Sysmem Fallback) and
# renders 50-100x slower WITH no error message -> 'none' is only promoted once the card has
# proven it has the room. Resolution order (the first one set wins):
# explicit UI/CLI choice > env CZ_OFFLOAD > config default_cpu_offload > auto.
OFFLOAD_CHOICES = ("auto", "none", "model", "sequential")
OFFLOAD_MODE = ((os.environ.get("CZ_OFFLOAD") or "").strip()
                or str(CONFIG.get("default_cpu_offload", "") or "").strip()).lower() or "auto"
if OFFLOAD_MODE not in OFFLOAD_CHOICES:
    _log(f"CZ_OFFLOAD/default_cpu_offload '{OFFLOAD_MODE}' unknown -> auto")
    OFFLOAD_MODE = "auto"
# The concrete mode resolved for 'auto' (set by _resolve_auto on the first load) and the
# flag of the runtime safety net (set by the VRAM callback during the denoise).
_AUTO_OFFLOAD = ""
_VRAM_DOWNGRADE = False

# Qwen-Image guidance. The UI's "guidance" slider = `true_cfg_scale` (a real CFG,
# which turns the negative prompt on). Advised range ~3-5 (default 4.0). The pipeline's
# distilled `guidance_scale` stays at 1.0 (see _cfg). A 0 inherited from a Z-Image config
# falls back to 4.0. Overridable through env QWEN_CFG.
GUIDANCE = float(os.environ.get("QWEN_CFG") or CONFIG.get("default_guidance") or 0) or 4.0

# Forced ratio (Fooocus-style) for upscale/img2img: when set, the INPUT image is
# center-cropped to that ratio before processing (crop to fit). Empty = the native ratio is
# preserved (the default). Format: 'W:H' or 'WxH' (e.g. '13:19', '832x1216'). Driven by the
# UI (checkbox + Aspect ratio dropdown) through set_force_ratio, or by config.txt
# 'force_upscale_ratio'.
FORCE_RATIO = (os.environ.get("CZ_FORCE_RATIO") or CONFIG.get("force_upscale_ratio") or "").strip()
# How to reach the forced ratio: 'crop' = center crop (loses the edges, the default),
# 'extend' = extends the image to the ratio by outpainting (loses nothing, adds generated
# bands). UI (radio) through set_force_ratio_mode, config 'force_ratio_mode'.
FORCE_RATIO_MODE = (os.environ.get("CZ_FORCE_RATIO_MODE")
                    or CONFIG.get("force_ratio_mode") or "crop").strip().lower()
# Seam-blending pass of the extend mode: after the bands are outpainted, a LIGHT img2img
# pass runs on the extended image and ONLY the bands plus a feathered transition margin are
# pasted back from it (the original centre stays untouched). 0 = off.
try:
    EXTEND_DENOISE = float(CONFIG.get("force_ratio_extend_denoise", 0.22) or 0.0)
except Exception:
    EXTEND_DENOISE = 0.22

# Sampler / scheduler. The Z-Image pipeline imposes a custom `sigmas` schedule: only
# the schedulers whose set_timesteps accepts `sigmas` work. In practice -> Euler
# flow-matching (native, the default), UniPC (multistep) and LCM flow-matching (interesting
# on distilled/Turbo models: few steps, guidance ~0-1).
# diffusers' DPM++ 2M / DPM2a / DPM++ SDE (dpmpp_sde) do NOT take custom sigmas ->
# incompatible (DPMSolverSDEScheduler also requires torchsde). Not exposed.
SAMPLER_CHOICES = ("euler", "unipc", "lcm")
SAMPLER = (os.environ.get("ZIMAGE_SAMPLER") or CONFIG.get("default_sampler") or "euler").strip().lower()
if SAMPLER not in SAMPLER_CHOICES:
    SAMPLER = "euler"

# Sigma schedule (= the "scheduler" in ComfyUI terms). sgm_uniform = Z-Image's native
# one (linspace + dynamic shift). beta/karras/exponential = a sigma remapping applied ON TOP
# of the pipeline's schedule (FlowMatchEuler/UniPC: use_*_sigmas). beta -> scipy.
SCHEDULE_CHOICES = ("sgm_uniform", "beta", "karras", "exponential")
# 'simple' (ComfyUI) names EXACTLY the native schedule exposed here as 'sgm_uniform': the
# default sigmas the pipeline hands the scheduler are linspace(1, 1/n, n), which is what
# ComfyUI calls 'simple' on a flow-matching model. Accepted as input everywhere
# (config/env/CLI/XYZ) so a CivitAI recipe can be copied word for word, but normalised to
# the canonical name: metadata and presets only ever carry one name.
_SCHEDULE_ALIASES = {"simple": "sgm_uniform"}
SCHEDULE_INPUTS = SCHEDULE_CHOICES + tuple(_SCHEDULE_ALIASES)   # listes ouvertes (CLI/XYZ)


def _norm_schedule(name, default="sgm_uniform"):
    """Nom de schedule -> nom canonique (alias resolus). Inconnu -> `default`."""
    n = (name or "").strip().lower()
    n = _SCHEDULE_ALIASES.get(n, n)
    return n if n in SCHEDULE_CHOICES else default


SCHEDULE = _norm_schedule(os.environ.get("ZIMAGE_SCHEDULE") or CONFIG.get("default_schedule"))
_SCHEDULE_FLAG = {"beta": "use_beta_sigmas", "karras": "use_karras_sigmas",
                  "exponential": "use_exponential_sigmas"}  # sgm_uniform -> no flag (native)
# The model's own scheduler config (captured on the first load) -> the base every other
# sampler is built from (keeps shift/flow params whatever the current sampler is).
_BASE_SCHED_CONFIG = None

# Hook de progression UI (gradio gr.Progress). None hors UI (CLI/serveur). Pose par
# the handlers through cz_pipeline._PROGRESS = ...
_PROGRESS = None
# Fooocus-style Stop: a global flag plus the interruption of the diffusers pipelines. Set
# by the handlers through cz_pipeline._STOP = ... and by request_stop().
_STOP = False

# GPU lock: serialises EVERY generation. Gradio does not serialise the events of
# different LISTENERS (manual Generate vs Run queue vs the detailer): two threads can then
# call the SAME shared pipeline and step the SAME scheduler -> its index runs past the end
# ("IndexError: index 31 is out of bounds for dimension 0 with size 31",
# scheduling_flow_match_euler_discrete.step). RLock: one thread's nested calls
# (txt2img_run -> generate, process_one -> _refine_whole) stay free.
_GPU_LOCK = threading.RLock()


def _gpu_serial(fn):
    """Decorator: runs fn under _GPU_LOCK (a single GPU generation at a time)."""
    import functools

    @functools.wraps(fn)
    def _locked(*args, **kwargs):
        with _GPU_LOCK:
            return fn(*args, **kwargs)
    return _locked

# Seed handling (Fooocus-style):
#  _LAST_SEED         = the CONCRETE seed of the last render (a random -1 is resolved to a
#                       real value) -> the "Reuse last seed" button + honest metadata.
#  _NO_SEED_INCREMENT = True -> a whole batch uses the same seed (no +i per image).
_LAST_SEED = -1
_NO_SEED_INCREMENT = False
# True -> in txt2img+upscale, ALSO save the original txt2img image (before the upscale).
_SAVE_PRE_UPSCALE = bool(CONFIG.get("save_pre_upscale", False))


def set_no_seed_increment(v):
    global _NO_SEED_INCREMENT
    _NO_SEED_INCREMENT = bool(v)


def set_save_pre_upscale(v):
    global _SAVE_PRE_UPSCALE
    _SAVE_PRE_UPSCALE = bool(v)



def set_guidance(g):
    global GUIDANCE
    GUIDANCE = float(g)


def _cfg(negative=None, guidance=None):
    """CFG kwargs shared by every Qwen pipeline: `true_cfg_scale` = the UI's guidance
    slider (a real CFG, which turns the negative prompt on), with the distilled
    `guidance_scale` fixed at 1.0.
    `guidance` = a per-call override (the edit protocol: a distilled model wants 1.0 while
    the global slider stays at 4.0)."""
    g = float(GUIDANCE) if guidance is None else float(guidance)
    # guidance_scale is no longer passed: Qwen-Image is not guidance-distilled,
    # diffusers ignores it (1.0 is its default already) and used to say so on every call.
    # true_cfg <= 1 = CFG off (Lightning/Rapid): the negative would have no effect, so it is
    # not passed rather than letting diffusers warn that it is ignored.
    return {"true_cfg_scale": g,
            "negative_prompt": (negative or None) if g > 1.0 else None}


# --- Prompt embedding cache -------------------------------------------------------
# Encoding a prompt sends the text encoder through the GPU. Under 'model' offload that
# transfer is paid on EVERY pipeline call -- including the detailer's passes, which redo
# the SAME prompt once per face and per hand.
# Measured on crispz-klein (9B GGUF, model offload): prompt+setup 5.2-6.1 s per pass
# without the cache against 1.7-1.8 s with it, for 0.3 s of diffusion. 2.1x per hand.
# Without offload the win drops to ~8 % (the encoder is already resident, nothing to move).
#
# encode_prompt() short-circuits the encoder as soon as it is handed its embeddings. So the
# TUPLE it returns is memorised and handed back to __call__ through _EMBED_OUTS.
# The tensors are kept in RAM (a few MB): they hold no VRAM and survive the offload's moves.
# QwenImage: encode_prompt -> (prompt_embeds, prompt_embeds_mask); the mask travels with
# the embeddings, __call__ wants both.
_EMBED_OUTS = ("prompt_embeds", "prompt_embeds_mask")
_EMBED_CACHE = {}
_EMBED_CACHE_MAX = max(0, int(CONFIG.get("prompt_embed_cache", 8) or 0))


def _embed_cache_clear(why=""):
    """Empties the cache. Called as soon as the encoder may have changed (base repo, VRAM
    release): an embedding computed by another encoder is wrong."""
    if _EMBED_CACHE:
        _dbg(f"prompt embed cache cleared ({len(_EMBED_CACHE)} entries){why}")
    _EMBED_CACHE.clear()


def _cached_prompt_embeds(pipe, prompt, kw):
    """Embeddings of `prompt` for this pipeline, computed once then reused.

    Returns a kwargs dict for __call__, or None when the cache is off, when the
    pipeline does not expose the expected API, or when the encoding fails: in all
    those cases the caller passes the prompt as text and nothing changes. A cache must
    never break a render.
"""
    if not _EMBED_CACHE_MAX or not _EMBED_OUTS:
        return None
    try:
        enc = getattr(pipe, "text_encoder", None)
        if enc is None or not hasattr(pipe, "encode_prompt"):
            return None
        # The LoRAs are part of the key: some of them touch the text encoder, and an
        # embedding computed without them would be wrong.
        # So is the replacement encoder: id(enc) alone is not enough, CPython recycles
        # the id of a freed object -- and another encoder encodes differently.
        key = (BASE_REPO, _TEXT_ENCODER_ACTIVE, id(enc), prompt,
               kw.get("max_sequence_length"),
               tuple(sorted((p, float(w)) for p, w in _APPLIED_LORAS)))
        hit = _EMBED_CACHE.get(key)
        if hit is None:
            out = pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)
            if not isinstance(out, (tuple, list)):
                out = (out,)
            hit = tuple(v.detach().to("cpu") if hasattr(v, "detach") else v
                        for v in out[:len(_EMBED_OUTS)])
            if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
                _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))      # FIFO, borne simple
            _EMBED_CACHE[key] = hit
            _dbg(f"prompt embeds computed and cached ({len(_EMBED_CACHE)}/"
                 f"{_EMBED_CACHE_MAX}) for {prompt[:40]!r}")
        else:
            _dbg(f"prompt embeds reused (text encoder not touched) for {prompt[:40]!r}")
        dev = pipe._execution_device
        return {name: (v.to(dev) if hasattr(v, "to") else v)
                for name, v in zip(_EMBED_OUTS, hit) if name}
    except Exception as e:
        _dbg(f"prompt embed cache off for this call ({type(e).__name__}: {e})")
        return None


def _encodes_with_image(pipe):
    """True when this pipeline's encode_prompt takes the image. The EDIT pipe
    (QwenImageEdit / EditPlus) prefixes the prompt with the image's vision tokens
    ('Picture 1: <|vision_start|><|image_pad|><|vision_end|>') and Qwen2.5-VL reads the image
    along with the instruction: an embedding computed without it is a text-only embedding.
    img2img / inpaint also receive `image`, but that is the starting image: their
    encode_prompt has no such parameter. An unreadable signature -> True, so no cache: one
    encoding too many costs time, a wrong encoding costs the image."""
    import inspect
    try:
        return "image" in inspect.signature(pipe.encode_prompt).parameters
    except (AttributeError, TypeError, ValueError):
        return True


def _qwen_call(pipe, **kw):
    """Calls a Qwen pipeline, tolerating diffusers API variations: when the installed
    version does not know `true_cfg_scale` / `negative_prompt`, those kwargs are dropped and
    the call is retried rather than crashing the generation."""
    # Reuse the embeddings if this prompt has already been encoded (see _EMBED_CACHE).
    # Passing them skips the text encoder: that is the whole win.
    # Except when the encoding depends on the image (the edit pipe): __call__ never
    # re-encodes when handed prompt_embeds, and the edit would run without its image.
    if (isinstance(kw.get("prompt"), str) and not any(k in kw for k in _EMBED_OUTS)
            and not (kw.get("image") is not None and _encodes_with_image(pipe))):
        _emb = _cached_prompt_embeds(pipe, kw["prompt"], kw)
        if _emb:
            kw.update(_emb)
            kw["prompt"] = None

    try:
        return pipe(**kw)
    except TypeError as e:
        # callback_on_step_end = the optional VRAM guard (see _vram_guard_kwargs):
        # an old diffusers build that does not know it runs without the guard.
        if any(k in kw for k in ("true_cfg_scale", "negative_prompt", "callback_on_step_end")):
            for k in ("true_cfg_scale", "negative_prompt", "callback_on_step_end"):
                kw.pop(k, None)
            _dbg(f"qwen call: retrying without the optional kwargs ({e})")
            return pipe(**kw)
        raise


def _scheduler_accepts_sigmas(sched):
    """The Z-Image pipeline calls set_timesteps(..., sigmas=<custom schedule>). A scheduler
    whose set_timesteps does not accept `sigmas` crashes at generation time."""
    import inspect
    try:
        return "sigmas" in inspect.signature(sched.set_timesteps).parameters
    except Exception:
        return False


def _build_scheduler(sampler, schedule, config):
    """Builds the chosen scheduler (sampler x schedule) from the model's native config.
    schedule (sgm_uniform/beta/karras/exponential) = a sigma remapping (use_*_sigmas)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    kw = {}
    flag = _SCHEDULE_FLAG.get((schedule or "").lower())
    if flag:
        kw[flag] = True
    name = (sampler or "euler").lower()
    if name == "unipc":
        from diffusers import UniPCMultistepScheduler
        try:
            return UniPCMultistepScheduler.from_config(config, use_flow_sigmas=True, **kw)
        except Exception:
            return UniPCMultistepScheduler.from_config(config, **kw)
    if name == "lcm":
        # LCM flow-matching: takes the pipeline's custom sigmas AND the schedule flags.
        # Falls back to Euler when the installed diffusers does not expose it.
        try:
            from diffusers import FlowMatchLCMScheduler
            return FlowMatchLCMScheduler.from_config(config, **kw)
        except Exception as e:
            _log(f"sampler 'lcm' unavailable ({e}); falling back to euler")
    return FlowMatchEulerDiscreteScheduler.from_config(config, **kw)


def _apply_sampler(pipe):
    """Applies the current scheduler (SAMPLER x SCHEDULE) to a pipe. Checks compatibility
    (custom sigmas) and falls back to Euler/sgm_uniform when it fails -> never a crash at
    generation time."""
    if _BASE_SCHED_CONFIG is None:
        return
    from diffusers import FlowMatchEulerDiscreteScheduler
    try:
        sched = _build_scheduler(SAMPLER, SCHEDULE, _BASE_SCHED_CONFIG)
        if not _scheduler_accepts_sigmas(sched):
            raise ValueError(f"{type(sched).__name__} does not accept the custom sigmas of Z-Image")
        pipe.scheduler = sched
        _dbg(f"sampler applied: {SAMPLER}/{SCHEDULE} -> {type(pipe.scheduler).__name__}")
    except Exception as e:
        _log(f"sampler '{SAMPLER}/{SCHEDULE}' incompatible ({e}); fallback Euler/sgm_uniform")
        try:
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(_BASE_SCHED_CONFIG)
        except Exception:
            pass


# Changing the scheduler is NOT a local change: it lives on the SHARED pipe. A denoise
# loop already running keeps its own `timesteps` list but steps whatever `pipe.scheduler`
# points at by then. A fresh scheduler knows nothing of those timesteps and has no
# begin_index, so diffusers looks the current timestep up and finds nothing:
#   IndexError: index 0 is out of bounds for dimension 0 with size 0
#   (scheduling_flow_match_euler_discrete._init_step_index -> index_for_timestep)
# Met on crispz-krea2 on 2026-09-28: checkpoint switched, then "Apply CivitAI recommended
# settings" (which sets the sampler AND the schedule), then Generate. Gradio does not
# serialise the events of DIFFERENT listeners -- the very hole _GPU_LOCK exists for, except
# that these two setters were never put under it.
# So the swap never lands under a running generation: free lock -> applied at once; held
# lock -> only recorded, and the next get_pipe() applies it. Every generation path goes
# through get_pipe() with the lock held.
_SAMPLER_DIRTY = False


def _reapply_sampler_all():
    """Re-applies the current scheduler to every cached pipe (base + derived). Returns
    False when a generation holds the GPU: the change is recorded, not applied."""
    global _SAMPLER_DIRTY
    # A try-acquire, not a wait: blocking here would freeze the dropdown handler for the
    # whole render (up to a 30-image batch). RLock -> a call from a thread that ALREADY
    # holds the lock (the job queue restoring a snapshot between two jobs) goes through and
    # applies at once, which is correct: that thread is between two generations.
    if not _GPU_LOCK.acquire(blocking=False):
        _SAMPLER_DIRTY = True
        _log(f"sampler/schedule {SAMPLER}/{SCHEDULE}: applied on the NEXT run "
             f"(a generation is running; swapping it now would crash that render)")
        return False
    try:
        _SAMPLER_DIRTY = False
        for p in [_BASE_PIPE] + list(_DERIVED.values()):
            if p is not None:
                _apply_sampler(p)
    finally:
        _GPU_LOCK.release()
    return True


def _apply_sampler_if_dirty():
    """Applies a sampler/schedule change that arrived while a generation was running.
    Called by get_pipe(), i.e. by every generation path, with _GPU_LOCK held."""
    global _SAMPLER_DIRTY
    if not _SAMPLER_DIRTY:
        return
    _SAMPLER_DIRTY = False
    _dbg(f"applying the deferred sampler/schedule {SAMPLER}/{SCHEDULE}")
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            _apply_sampler(p)


def _sampler_status():
    """The label shown next to the two dropdowns. Says so when the change is only
    recorded: telling the user it is active while the render still uses the old one is
    exactly the confusion to avoid."""
    return (f"Sampler: {SAMPLER} / {SCHEDULE}"
            + (" — on the next run" if _SAMPLER_DIRTY else ""))


def set_sampler(name):
    """Changes the sampler (euler/unipc) and re-applies it to the cached pipes (no
    reload). No effect on the Omni pipe (its own scheduler)."""
    global SAMPLER
    name = (name or "euler").strip().lower()
    if name not in SAMPLER_CHOICES:
        name = "euler"
    if name != SAMPLER:
        SAMPLER = name
        _log(f"sampler -> {SAMPLER}")
        _reapply_sampler_all()
    return _sampler_status()


def set_schedule(name):
    """Changes the sigma schedule (sgm_uniform/beta/karras/exponential, alias 'simple'
    = sgm_uniform) and re-applies it to the cached pipes."""
    global SCHEDULE
    name = _norm_schedule(name)
    if name != SCHEDULE:
        SCHEDULE = name
        _log(f"schedule -> {SCHEDULE}")
        _reapply_sampler_all()
    return _sampler_status()


def _progress(frac, desc=""):
    if _PROGRESS is not None:
        try:
            _PROGRESS(min(1.0, max(0.0, float(frac))), desc)
        except Exception:
            pass


# ---- Model loading feedback (terminal + UI) ----
# from_pretrained is blocking and silent (the first load downloads from HF -> several
# minutes). So the load runs in a thread and every ~2s a terminal line plus the Gradio bar
# are refreshed (elapsed time + allocated VRAM). Config block "load_progress";
# enabled=false -> a direct load (no thread, zero cost).
_LOAD_CFG = CONFIG.get("load_progress") if isinstance(CONFIG.get("load_progress"), dict) else {}
LOAD_PROGRESS_ENABLED = bool(_LOAD_CFG.get("enabled", True))
_LOAD_TARGET_GB = float(_LOAD_CFG.get("target_vram_gb", 14.0))
_LOAD_HEARTBEAT = float(_LOAD_CFG.get("heartbeat_s", 2.0))


def _fmt_load(label, elapsed, vram_gb):
    """Loading progress text (pure, testable). VRAM > 0 -> the loading-into-memory
    phase; otherwise the download/disk-read phase."""
    if vram_gb > 0.05:
        return f"{label}... {elapsed:.0f}s | {vram_gb:.1f} GB in VRAM"
    return f"{label}... {elapsed:.0f}s (downloading / reading, first run only)"


def _load_pct(elapsed, vram_gb, target_gb=None):
    """An honest %: based on the allocated VRAM / target once the load into memory has
    started (capped at 0.95); during the download (VRAM~0) a small time-based bar."""
    target_gb = target_gb or _LOAD_TARGET_GB
    if vram_gb <= 0.05:
        return min(0.12, elapsed / 600.0)
    return min(0.95, vram_gb / max(1.0, float(target_gb)))


def _load_monitor(label, fn):
    """Runs fn() (a blocking load) in a thread and refreshes terminal + UI (time +
    VRAM) every ~2s. Returns fn's result (re-raises its exception)."""
    if not LOAD_PROGRESS_ENABLED:
        return fn()
    box = {}

    def _work():
        try:
            box["v"] = fn()
        except BaseException as e:   # noqa: BLE001 - it is re-raised in the main thread
            box["e"] = e

    th = threading.Thread(target=_work, daemon=True)
    t0 = time.time()
    th.start()
    while True:
        th.join(timeout=_LOAD_HEARTBEAT)
        el = time.time() - t0
        vram = (torch.cuda.memory_allocated() / 1024 ** 3) if DEVICE == "cuda" else 0.0
        line = _fmt_load(label, el, vram)
        if cz_core.LOG_LEVEL >= 1:
            sys.stderr.write("\r[crispz][load] " + line + "        ")
            sys.stderr.flush()
        _progress(_load_pct(el, vram), "Loading " + line)
        if not th.is_alive():
            break
    if cz_core.LOG_LEVEL >= 1:
        sys.stderr.write("\n")
        sys.stderr.flush()
    if "e" in box:
        raise box["e"]
    return box.get("v")


def request_stop():
    """Asks for a stop: halts the running denoise loop (pipe._interrupt) and the
    batch/tile loops (_STOP). Near-immediate (it stops at the next step)."""
    global _STOP
    _STOP = True
    n = 0
    for p in [_BASE_PIPE] + list(_DERIVED.values()):
        if p is not None:
            try:
                p._interrupt = True
                n += 1
            except Exception:
                pass
    _log(f"STOP requested (interrupt set on {n} pipeline(s))")
    return "Stopping..."


def set_zimage_model(repo_or_path):
    """Changes the Qwen model. An HF repo / diffusers folder -> BASE_REPO.
    A single-file checkpoint (a Civitai .safetensors, a .gguf) -> a transformer override."""
    global BASE_REPO, ZIMAGE_TRANSFORMER
    if not repo_or_path:
        return
    if _is_single_file(repo_or_path):
        # A transformer-only change: NO free_vram -> _ensure_base will swap the
        # transformer alone (VAE + text encoder kept in VRAM).
        if repo_or_path != ZIMAGE_TRANSFORMER:
            ZIMAGE_TRANSFORMER = repo_or_path
            _log("Qwen transformer (single-file) changed -> transformer swap on next run")
    elif repo_or_path != BASE_REPO:
        # The base repo changes: the VAE/encoder/tokenizer change too -> a full reload.
        BASE_REPO = repo_or_path
        free_vram()
        _log("Qwen base repo changed -> will reload")


def set_zimage_transformer(path):
    """Sets (or removes with '' / None) the single-file transformer.

    Does NOT release the pipeline: with the same base repo, _ensure_base will only reload
    the transformer (_swap_transformer) and keep VAE + text encoder in VRAM.
"""
    global ZIMAGE_TRANSFORMER
    path = path or None
    if path != ZIMAGE_TRANSFORMER:
        ZIMAGE_TRANSFORMER = path
        _log(f"Qwen transformer -> {path or '(repo de base)'} "
             "-> transformer swap on next run (base components kept)")


# --- Replacement text encoder ---------------------------------------------------------
# Qwen-Image reads the LAST hidden state of the encoder (hidden_states[-1]) in txt_norm +
# txt_in, both 3584 wide (joint_attention_dim): an encoder only fits if it has the same
# family (qwen2_5_vl -- the edit pipe shows it the image), the same width and the same number
# of layers as the one it replaces. An "abliterated" or fine-tuned Qwen2.5-VL-7B plugs in as
# is. That is checked on the config, BEFORE reading 16 GB -- and for EACH pipe against ITS
# repo (the base one, or the edit repo).
_SINGLE_FILE_EXTS = (".safetensors", ".ckpt", ".pt", ".sft", ".gguf")


def _looks_single_file(p):
    """True when the NAME is that of a single-file checkpoint, whether it exists or not
    (_is_single_file also requires the file to be there)."""
    return bool(p) and str(p).lower().endswith(_SINGLE_FILE_EXTS)


def _split_hf_src(src):
    """'owner/repo/sub/folder' -> ('owner/repo', 'sub/folder'). The weights of an encoder
    published on HF often sit in a subfolder of the repo."""
    parts = [p for p in str(src).replace("\\", "/").split("/") if p]
    if len(parts) > 2:
        return "/".join(parts[:2]), "/".join(parts[2:])
    return str(src), None


def _enc_dims(cfg):
    """(width, layers, family) of a transformers config. Qwen2.5-VL keeps the text part
    under 'text_config' (on top of the top level since transformers 4.53, and only there in
    recent configs); T5 says d_model / num_layers."""
    c = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    h = c.get("hidden_size") or c.get("d_model")
    n = c.get("num_hidden_layers") or c.get("num_layers")
    return (int(h) if h else None, int(n) if n else None, cfg.get("model_type"))


def _base_text_encoder_config(base=None):
    """The config.json of repo `base`'s encoder (BASE_REPO by default), or None when
    unreadable."""
    base = (base or BASE_REPO or "").strip()
    try:
        cfg = os.path.join(base, "text_encoder", "config.json")
        if not os.path.isfile(cfg):
            from huggingface_hub import hf_hub_download
            try:
                cfg = hf_hub_download(base, "text_encoder/config.json", local_files_only=True)
            except Exception:
                cfg = hf_hub_download(base, "text_encoder/config.json")
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _dbg(f"cannot read {base}'s text encoder config: {e}")
        return None


def _text_encoder_source(src):
    """Locates the encoder `src`: (config, folder or repo, subfolder) or None.
    A local folder: config.json at the root or inside text_encoder/. An HF repo: the same,
    or the subfolder named in the id."""
    src = (src or "").strip()
    if not src:
        return None
    if os.path.isdir(src):
        for sub in (None, "text_encoder"):
            p = os.path.join(src, sub, "config.json") if sub else os.path.join(src, "config.json")
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        return json.load(f), src, sub
                except Exception:
                    return None
        return None
    if os.path.exists(src) or _looks_single_file(src) or "\\" in src or os.path.isabs(src):
        return None
    repo, sub0 = _split_hf_src(src)
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None
    for sub in ([sub0] if sub0 else [None, "text_encoder"]):
        rel = f"{sub}/config.json" if sub else "config.json"
        for local in (True, False):          # the cache first: works offline
            try:
                p = hf_hub_download(repo, rel, local_files_only=local)
                with open(p, encoding="utf-8") as f:
                    return json.load(f), repo, sub
            except Exception:
                continue
    return None


def _encoder_label(src):
    """A readable name for an encoder: the FOLDER's name -- never the path, which would
    end up in shared PNGs along with the Windows session name -- or the HF repo id."""
    src = (src or "").strip()
    if not src:
        return ""
    if os.path.isabs(src) or os.path.exists(src) or "\\" in src:
        parts = [p for p in src.replace("\\", "/").split("/") if p]
        if len(parts) >= 2 and parts[-1] == "text_encoder":
            return parts[-2]
        return parts[-1] if parts else src
    return src


def _text_encoder_problem(src, base=None):
    """Why `src` should be refused as the encoder of repo `base` (BASE_REPO by default),
    or None when it fits."""
    src = (src or "").strip()
    if not src:
        return None
    if src.lower().endswith(".gguf"):
        return ("a GGUF text encoder is a ComfyUI / llama.cpp file; this app loads the "
                "transformers folder (config.json + .safetensors)")
    if os.path.isfile(src) or _looks_single_file(src):
        return ("a single file carries no config.json; point to the FOLDER that holds "
                "config.json and the weights")
    found = _text_encoder_source(src)
    if found is None:
        return ("no config.json found, neither at its root nor in text_encoder/"
                if os.path.isdir(src) else
                "neither a folder on this machine nor a readable Hugging Face repo")
    ref_cfg = _base_text_encoder_config(base)
    if ref_cfg is None:
        return None                      # nothing to compare: the load will decide
    (h, n, t), (rh, rn, rt) = _enc_dims(found[0]), _enc_dims(ref_cfg)
    b = (base or BASE_REPO)
    if t and rt and t != rt:
        return f"a '{t}' model, and {b} uses a '{rt}' text encoder"
    if h and rh and h != rh:
        return (f"hidden size {h}, and {b}'s encoder is {rh} wide: the transformer "
                f"cannot read its embeddings")
    if n and rn and n != rn:
        return f"{n} layers, and {b}'s encoder has {rn}"
    return None


def _encoder_class(base=None):
    """The encoder's transformers class, read from repo `base`'s model_index.json
    (Qwen2_5_VLForConditionalGeneration here): the same one diffusers would have loaded."""
    base = (base or BASE_REPO or "").strip()
    try:
        p = os.path.join(base, "model_index.json")
        if not os.path.isfile(p):
            from huggingface_hub import hf_hub_download
            try:
                p = hf_hub_download(base, "model_index.json", local_files_only=True)
            except Exception:
                p = hf_hub_download(base, "model_index.json")
        with open(p, encoding="utf-8") as f:
            lib, cls = json.load(f)["text_encoder"]
        import importlib
        return getattr(importlib.import_module(lib), cls)
    except Exception as e:
        raise RuntimeError(f"cannot tell which class {base}'s text encoder uses "
                           f"({type(e).__name__}: {e})") from e


def _load_text_encoder(src, base=None):
    """Loads the encoder `src` in DTYPE, with repo `base`'s class."""
    found = _text_encoder_source(src)
    if found is None:
        raise RuntimeError(f"{src}: no config.json")
    _cfg, where, sub = found
    kw = {"torch_dtype": DTYPE}
    if sub:
        kw["subfolder"] = sub
    return _encoder_class(base).from_pretrained(where, **kw)


def _pick_text_encoder(base, which):
    """A replacement encoder for the `which` pipe ('base' / 'edit') of repo `base`:
    (model, TEXT_ENCODER) when it suits THAT repo and loads, otherwise (None, '') with a log
    line -- the pipe then keeps its repo's encoder. Never raises: an option must not bring a
    generation down."""
    src = TEXT_ENCODER
    if not src:
        return None, ""
    label = _encoder_label(src)
    try:
        why = _text_encoder_problem(src, base)
    except Exception as e:
        why = f"its config could not be checked ({type(e).__name__}: {e})"
    enc = None
    if not why:
        try:
            enc = _load_monitor(f"text encoder {label}", lambda: _load_text_encoder(src, base))
        except Exception as e:
            why = f"it failed to load ({type(e).__name__}: {e})"
    if why:
        _log(f"text encoder {label} NOT used on the {which} pipe: {why}. {base}'s own "
             f"encoder runs instead; the image metadata says so (text_encoder_not_applied).")
        return None, ""
    _log(f"text encoder: {label} replaces {base}'s own on the {which} pipe "
         f"(tokenizer, processor, VAE and transformer unchanged)")
    return enc, src


def list_text_encoders():
    """Encoder folders offered in the Models tab: the subfolders with a config.json inside
    `text_encoders_dir`, or inside text_encoders / text_encoder / clip next to the
    checkpoints folder or its parent (ComfyUI and Forge conventions)."""
    roots = [TEXT_ENCODERS_DIR] if TEXT_ENCODERS_DIR else []
    here = os.path.abspath(CHECKPOINTS_DIR or ".")
    for up in (os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        roots += [os.path.join(up, n) for n in ("text_encoders", "text_encoder", "clip")]
    out = []
    for r in roots:
        try:
            names = sorted(os.listdir(r))
        except OSError:
            continue
        for d in names:
            p = os.path.join(r, d)
            if p in out or not os.path.isdir(p):
                continue
            if (os.path.isfile(os.path.join(p, "config.json"))
                    or os.path.isfile(os.path.join(p, "text_encoder", "config.json"))):
                out.append(p)
    return out



def _hf_cache_dir():
    """The Hugging Face cache folder (follows HF_HUB_CACHE / HF_HOME), or None."""
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except Exception:
        return None


def _scan_cached_encoders():
    """[(HF id, config)] from the Hugging Face cache: repos that are NOT diffusers
    pipelines (no model_index.json), one of whose configs -- at the root or in a subfolder --
    has its weights next to it (the most recent revision)."""
    root = _hf_cache_dir()
    if not root or not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        if not d.startswith("models--"):
            continue
        repo = d[len("models--"):].replace("--", "/", 1)
        snaps = os.path.join(root, d, "snapshots")
        try:
            revs = sorted(os.listdir(snaps),
                          key=lambda r: os.path.getmtime(os.path.join(snaps, r)), reverse=True)
        except OSError:
            continue
        if not revs:
            continue
        snap = os.path.join(snaps, revs[0])
        if os.path.isfile(os.path.join(snap, "model_index.json")):
            continue                                 # a diffusers pipeline, not an encoder
        try:
            subs = [""] + sorted(s for s in os.listdir(snap) if os.path.isdir(os.path.join(snap, s)))
        except OSError:
            continue
        for s in subs:
            p = os.path.join(snap, s) if s else snap
            try:
                with open(os.path.join(p, "config.json"), encoding="utf-8") as f:
                    cfg = json.load(f)
                if not isinstance(cfg, dict):
                    continue
                if not any(fn.endswith(".safetensors") for fn in os.listdir(p)):
                    continue                         # the config alone, the weights are not downloaded
            except Exception:
                continue
            out.append((f"{repo}/{s}" if s else repo, cfg))
    return out


def list_cached_text_encoders(base=None):
    """COMPATIBLE encoders already downloaded in the Hugging Face cache, as (name, HF id).
    An encoder downloaded from HF lives in that cache, not in a text_encoders folder:
    without this sweep the Models tab list did not show it (caught on klein on 2026-09-10).
    Compatible = the same family, width and number of layers as the base repo's encoder. The
    value is the HF id: readable in the metadata."""
    ref_cfg = _base_text_encoder_config(base)
    if not ref_cfg:
        return []
    ref = _enc_dims(ref_cfg)
    return [(hid, hid) for hid, cfg in _scan_cached_encoders() if _enc_dims(cfg) == ref]


def cached_text_encoder_mismatches(base=None):
    """Encoders in the HF cache of the SAME family but of another size than the base
    repo's: hidden from the list (they would be refused), named next to it so one knows why.
    ([(HF id, width)], expected width)."""
    ref_cfg = _base_text_encoder_config(base)
    if not ref_cfg:
        return [], None
    rh, rn, rt = _enc_dims(ref_cfg)
    out = []
    for hid, cfg in _scan_cached_encoders():
        h, n, t = _enc_dims(cfg)
        if t == rt and (h, n) != (rh, rn):
            out.append((hid, h))
    return out, rh


def set_text_encoder(src):
    """Picks the text encoder ('' = the repo's). A change RELEASES both pipelines, base
    and edit -- the encoder loads with them, no hot swap under the offload hooks -- and
    free_vram empties the embedding cache, which the previous one computed."""
    global TEXT_ENCODER
    src = (src or "").strip()
    if src == TEXT_ENCODER:
        return
    TEXT_ENCODER = src
    free_vram()
    _log(f"text encoder -> {_encoder_label(src) or '(repo default)'} -> full reload of "
         f"the base and edit pipes on next run")


def _safetensors_header(path):
    """The JSON header of a .safetensors (tensor names/dtypes/shapes, NEVER the
    weights) -- a few hundred KB read at most, even on a 12 GB file."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(min(n, 10_000_000)).decode("utf-8", "ignore"))


def _safetensors_unsupported(path):
    """Returns a reason (str) when the .safetensors is NOT loadable, otherwise None.
    Only reads the header (fast). Two cases stay unsupported:
      - a LoRA file filed in the checkpoints folder (kohya/peft keys)
      - SVDQuant / Nunchaku (tensors named '*.qweight'): pre-quantized INT4 weights that
        require the nunchaku runtime (dedicated kernels), not dequantizable here.
    ComfyUI-style 'scaled' FP8 / INT8 are NO LONGER rejected: they go through the dequant
    loader (_safetensors_dequant + _load_dequant_state_dict).
"""
    try:
        hdr = _safetensors_header(path)
        has_qweight = False
        lora_keys = 0
        te_keys = 0
        dit_keys = 0
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            if k.endswith(".qweight"):
                has_qweight = True
            # A Qwen2.5-VL text encoder (the ComfyUI file
            # 'qwen_2.5_vl_7b_fp8_scaled'): LLM layers + a vision tower, never any
            # diffusion block.
            if k.startswith(("model.layers.", "visual.", "lm_head.", "model.embed_tokens",
                             "language_model.", "model.language_model.")):
                te_keys += 1
            if "transformer_blocks" in k or k.startswith(("img_in", "txt_in")):
                dit_keys += 1
            if (".lora_down." in k or ".lora_up." in k or ".lora_A." in k
                    or ".lora_B." in k or k.startswith(("lora_unet_", "lora_te"))):
                lora_keys += 1
        # A LoRA file filed in the checkpoints folder (a classic mistake): loading it
        # as a transformer sends diffusers looking for a default config (SD1.5) -> a 404
        # 'stable-diffusion-v1-5 does not appear to have a file named config.json'.
        if lora_keys >= 4:
            return "LoRA file, not a checkpoint - move it to the LoRA folder and pick it in Models > LoRA"
        # A text encoder filed with the checkpoints (a Civitai 'text encoder'
        # download): it is not an image model, dequantizing it would waste ~15 GB of cache
        # and loading it as a transformer would fail. The pipe takes its encoder from the
        # base repo.
        if te_keys >= 4 and dit_keys == 0:
            return ("text encoder (Qwen2.5-VL), not an image model - the pipeline takes its "
                    "text encoder from the base repo; nothing to do with this file")
        # '*.qweight' = pre-quantized weights (SVDQuant/Nunchaku, GPTQ-like). A clear signal:
        # a normal BF16/FP16 checkpoint never has a 'qweight'.
        if has_qweight:
            return "SVDQuant/Nunchaku INT4"
    except Exception:
        pass
    return None


def _safetensors_dequant(path):
    """Returns the ComfyUI quantization scheme to dequantize at load time
    ('FP8', 'FP8 scaled' or 'INT8 scaled'), otherwise None (BF16/FP16 -> the normal path).
    The ComfyUI 'scaled' format seen on Civitai checkpoints:
      X.weight (F8_E4M3 or I8) + X.weight_scale (F32, scalar or per row [out,1])
      + X.comfy_quant (a small U8 descriptor blob, to be dropped).
    NB: an AIO bundle whose text encoder ALONE is quantized (BF16 transformer) also
    triggers -> the dequant loader filters the transformer and leaves it untouched.
    U8 alone does not trigger: 'comfy_quant' blobs are U8 in healthy files.
"""
    try:
        hdr = _safetensors_header(path)
        has_fp8 = has_int = has_scale = False
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            dt = str(v.get("dtype", "")).upper()
            if dt.startswith("F8"):
                has_fp8 = True
            elif dt in ("I8", "I4", "U4", "INT8"):
                has_int = True
            if k.endswith(("weight_scale", "scale_weight")):
                has_scale = True
        if has_fp8:
            return "FP8 scaled" if has_scale else "FP8"
        if has_int and has_scale:
            return "INT8 scaled"
    except Exception:
        pass
    return None


# Key markers of the Qwen-Image transformer (original layout OR ComfyUI prefix): used
# by the dequant loader to refuse a quantized checkpoint of ANOTHER architecture (it would
# load incoherent weights).
_QWEN_KEY_MARKERS = ("transformer_blocks.", "img_in", "txt_in", "time_text_embed")

# The ComfyUI/LDM prefix of single-file diffusion checkpoints. diffusers 0.39 maps
# QwenImageTransformer2DModel with an IDENTITY function (no key conversion at all): the state
# dict must therefore arrive IN THE DIFFUSERS LAYOUT, prefix stripped. Otherwise every key is
# "unexpected", no weight is loaded, the model stays on 'meta' and dispatch_model breaks on
# "Cannot copy out of meta tensor; no data!".
_COMFY_PREFIX = "model.diffusion_model."


# ----------------------------------------------------------------------------
# Disk cache of dequantized transformers (ComfyUI FP8/INT8 -> bf16), carried over
# from crispz-studio. A dequant reads and converts the whole file (minutes on a
# HDD); the bf16 is written ONCE here and later loads become a normal
# single-file one (seconds). The KEY is the ORIGINAL file (path+size+mtime):
# deleting this cache is always safe, it rebuilds on demand.
# ----------------------------------------------------------------------------
import hashlib as _dqhash

_DQ_CACHE_CFG = str(CONFIG.get("dequant_cache", "auto") or "auto").strip()
try:
    DEQUANT_CACHE_MAX_GB = float(CONFIG.get("dequant_cache_max_gb", 60) or 0)
except Exception:
    DEQUANT_CACHE_MAX_GB = 60.0


def _file_key(path):
    """A file's stable, cheap identity: (absolute path, size, mtime)."""
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, int(st.st_mtime))


def _dequant_cache_dir():
    """The dequant cache folder, created on demand. None = cache disabled."""
    if _DQ_CACHE_CFG.lower() in ("off", "none", "0", "false"):
        return None
    d = (os.path.join(HERE, "cache", "dequant")
         if _DQ_CACHE_CFG.lower() in ("auto", "") else _DQ_CACHE_CFG)
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except Exception as e:
        _dbg(f"dequant cache dir unavailable ({e})")
        return None


# Range of each 8-bit format: the largest stored value a QUANTIZED weight
# (weight / scale) can reach.
_QUANT_RANGE = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0, torch.int8: 127.0}


def _stored_at_scale(t, s, qdtype, cfg=None):
    """True when the stored weights are ALREADY at their real scale, a weight_scale being
    supplied on top -- not to be applied. Carried over from crispz-klein 1.34.1.

    A normal 'scaled' FP8 stores weight / scale: it FILLS the format's range (448 in E4M3)
    and max|stored| / (scale x range) is 1 / scale (71 to 1,691 across the 16 FP8/INT8 files
    of the library). kleinFinalcutFP16FP8_comfyQuant stores its weights as they are (0.375
    out of 448) and supplies amax / 448 anyway: ratio 1.03. Applying the scale made every
    weight 1,200 to 1,700 times too small, and the image came out as noise. MX scales
    (uint8 = an E8M0 exponent) are never concerned.
"""
    rng = _QUANT_RANGE.get(qdtype)
    fmt = str((cfg or {}).get("format", "")).lower()
    if rng is None or s.dtype == torch.uint8 or fmt.startswith("mx"):
        return False
    smax = float(s.detach().float().abs().max())
    if smax <= 0.0:
        return False
    amax = float(t.detach().float().abs().max())
    # 1. the range is barely used (a normal file fills it)...
    if amax >= rng / 4:
        return False
        # 2. ... AND the scale describes exactly the stored values: ratio ~1.
    ratio = amax / (smax * rng)
    return 0.5 <= ratio < 2.0


_PRESCALED = {}      # file key -> bool (read once per file and per session)


def _source_prescaled(src):
    """Does the source file store its weights already at scale? Read on the SMALLEST
    quantized tensor that carries a scale: a few KB to read. Any error = False: the cache key
    then does not change."""
    try:
        fk = _file_key(src)
    except OSError:
        return False
    if fk in _PRESCALED:
        return _PRESCALED[fk]
    res = False
    try:
        hdr = _safetensors_header(src)
        best = None
        for k, v in hdr.items():
            if not (isinstance(v, dict) and k.endswith(".weight")):
                continue
            if str(v.get("dtype", "")).upper() not in ("F8_E4M3", "F8_E5M2", "I8"):
                continue
            sk = next((c for c in (k + "_scale", k[:-len(".weight")] + ".scale_weight")
                       if c in hdr), None)
            if sk is None:
                continue
            n = 1
            for d in v.get("shape") or [1]:
                n *= int(d)
            if best is None or n < best[0]:
                best = (n, k, sk)
        if best:
            from safetensors import safe_open
            with safe_open(src, framework="pt", device="cpu") as f:
                t = f.get_tensor(best[1])
                s = f.get_tensor(best[2])
            res = _stored_at_scale(t, s, t.dtype)
    except Exception as e:
        _dbg(f"prescaled check failed on {os.path.basename(src)}: {e}")
        res = False
    _PRESCALED[fk] = res
    return res


def _dequant_cache_path(src, legacy=False):
    """Path of the cached bf16 for a source checkpoint. The key includes size+mtime: a
    replaced file (same name) never reuses the old cache."""
    d = _dequant_cache_dir()
    if not d:
        return None
    try:
        p, size, mtime = _file_key(src)
    except OSError:
        return None
    # A file already stored at scale changes key: its old cache was written by the
    # loader that applied the scale wrongly. legacy=True returns the old key.
    tag = "bf16-v2"
    if not legacy and _source_prescaled(src):
        tag = "bf16-v2-prescaled"
    h = _dqhash.sha1(
        f"{p.lower()}|{size}|{mtime}|{tag}".encode("utf-8")).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(src))[0][:48]
    return os.path.join(d, f"{base}.{h}.safetensors")


def _dequant_cache_prune(keep=None):
    """Caps the cache (dequant_cache_max_gb, 0 = unlimited): deletes the least recently
    USED files (atime, else mtime) until it is back under the threshold."""
    d = _dequant_cache_dir()
    if not d or DEQUANT_CACHE_MAX_GB <= 0:
        return
    try:
        files = []
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if not f.endswith(".safetensors") or not os.path.isfile(fp):
                continue
            st = os.stat(fp)
            files.append((max(st.st_atime, st.st_mtime), st.st_size, fp))
        total = sum(s for _t, s, _p in files)
        cap = DEQUANT_CACHE_MAX_GB * 1024**3
        for _t, size, fp in sorted(files):          # plus ancien acces d'abord
            if total <= cap:
                break
            if keep and os.path.abspath(fp) == os.path.abspath(keep):
                continue
            try:
                os.remove(fp)
                total -= size
                _log(f"dequant cache: evicted {os.path.basename(fp)} "
                     f"({size / 1024**3:.1f} GB, over the {DEQUANT_CACHE_MAX_GB:.0f} GB cap)")
            except OSError as e:
                _dbg(f"dequant cache evict failed {fp}: {e}")
    except Exception as e:
        _dbg(f"dequant cache prune failed: {e}")


def _dequant_cache_store(src, sd):
    """Writes the dequantized state dict into the cache (best effort: any error is
    ignored, the current load already has the dict in memory). Atomic write through a
    renamed .tmp -> an interruption never leaves a truncated cache."""
    dst = _dequant_cache_path(src)
    if not dst:
        return
    try:
        from safetensors.torch import save_file
        t0 = time.time()
        tmp = dst + ".tmp"
        # contiguous(): safetensors refuses non-contiguous views (they come from the
        # dequant slices); an implicit clone, and we are in RAM already.
        save_file({k: v.contiguous() for k, v in sd.items()}, tmp)
        os.replace(tmp, dst)
        gb = os.path.getsize(dst) / 1024**3
        _log(f"dequant cache: saved {gb:.1f} GB in {time.time() - t0:.1f}s "
             f"-> next load of this checkpoint skips the dequant")
        # THIS file's old cache, written under the old key (wrong weights, see
        # _dequant_cache_path): it has been replaced, so it is deleted.
        old = _dequant_cache_path(src, legacy=True)
        if old and os.path.abspath(old) != os.path.abspath(dst) and os.path.isfile(old):
            try:
                ogb = os.path.getsize(old) / 1024**3
                os.remove(old)
                _log(f"dequant cache: removed the stale {os.path.basename(old)} "
                     f"({ogb:.1f} GB, written by the loader that applied the scale twice)")
            except OSError as e:
                _dbg(f"stale dequant cache not removed {old}: {e}")
        _dequant_cache_prune(keep=dst)
    except Exception as e:
        _log(f"dequant cache: not saved ({e})")
        try:
            os.remove(dst + ".tmp")
        except OSError:
            pass


def _hadamard_ortho(n):
    """The comfy-quants ConvRot 'regular hadamard' matrix -- CAREFUL, this is NOT
    Sylvester's construction: the base is that precise H4, extended by Kronecker products
    up to n (a power of 4), then normalised by 1/sqrt(n). Orthonormal AND symmetric -> the
    reconstruction simply multiplies by the same matrix again.
    (Checked against src/comfy_quants/formats/convrot.py; with a Sylvester the correlation
    to the base weights drops to ~0 -> pure noise.)
"""
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]])
    H = h4
    while H.shape[0] < n:
        H = torch.kron(H, h4)
    if H.shape[0] != n:
        raise ValueError(f"convrot groupsize {n} is not a power of 4")
    return H / (float(n) ** 0.5)


def _safetensors_comfy_prefixed(path):
    """True when the .safetensors is in the ComfyUI layout ('model.diffusion_model.*').
    Only reads the header. Such a file can NOT go as is into from_single_file (identity
    mapping on the diffusers side, see _COMFY_PREFIX)."""
    try:
        return any(k.startswith(_COMFY_PREFIX)
                   for k in _safetensors_header(path) if k != "__metadata__")
    except Exception:
        return False


def _load_dequant_state_dict(path):
    """Loads a ComfyUI single-file into RAM and returns it IN THE DIFFUSERS LAYOUT,
    dequantized to DTYPE (bf16) tensor by tensor. Serves both cases: quantized (FP8/INT8
    'scaled') and merely PREFIXED (bf16/fp16 -- nothing to dequantize, just the prefix to
    strip):
      - an AIO bundle (transformer + text encoder + VAE): only the
        'model.diffusion_model.*' keys are kept (VAE + encoder = the base repo's);
      - X.weight (F8/I8) * X.weight_scale (scalar or per row) -> bf16;
      - the X.comfy_quant blob: when 'convrot' is declared (ComfyUI int8_tensorwise), the
        grouped Hadamard rotation (256 by default) is UNDONE after the descale -- without
        that the weights are pure noise;
      - the quantization keys (weight_scale/scale_weight, comfy_quant, the scaled_fp8
        marker) are consumed and dropped.
    The resulting dict goes into from_single_file (diffusers key conversion included).
    VRAM/RAM note: dequantized = the footprint of a full BF16; FP8 only saves disk and
    download, not memory.
"""
    from safetensors import safe_open
    t0 = time.time()
    hdr = _safetensors_header(path)
    entries = [(k, v) for k, v in hdr.items()
               if k != "__metadata__" and isinstance(v, dict)]
    # AIO bundle: keep only the transformer. (No ComfyUI prefix = a transformer-only
    # file in the original layout -> no filtering.) crispz-krea2's method: the prefix is
    # stripped at READ time, so everything downstream (scales, qcfg, architecture guard,
    # returned state dict) works on diffusers-layout keys, with no variant.
    prefix = ""
    if any(k.startswith(_COMFY_PREFIX) for k, _ in entries):
        prefix = _COMFY_PREFIX
        entries = [(k, v) for k, v in entries if k.startswith(prefix)]
    # Architecture guard: a quantized checkpoint of ANOTHER model (keys without a
    # single Qwen-Image marker) would load incoherent weights -> a clear refusal.
    if not any(any(m in k[len(prefix):] for m in _QWEN_KEY_MARKERS) for k, _ in entries):
        raise RuntimeError(
            f"{os.path.basename(path)}: quantized checkpoint does not look like a "
            "Qwen-Image transformer (different architecture); this build only loads "
            "Qwen-Image models.")
    # SEQUENTIAL read in the file's PHYSICAL order (data_offsets): a HDD collapses on
    # random access, and the key order does not follow the data's.
    entries.sort(key=lambda kv: kv[1].get("data_offsets", [0])[0])
    raw = {}
    qcfg = {}
    # comfy-quants declares the scheme either in PER-TENSOR blobs (X.comfy_quant), or
    # CENTRALLY in __metadata__._quantization_metadata (the StableYogi variant:
    # {"layers": {"blocks...": {"format": "int8_tensorwise", "convrot": true,
    # "convrot_groupsize": 256}}}). Ignoring that variant leaves the rotation in place ->
    # weights as pure noise (observed on the Krea 2 INT8s; the same format is possible
    # here). The per-tensor blobs win.
    try:
        qm = json.loads((hdr.get("__metadata__") or {}).get(
            "_quantization_metadata") or "{}")
        for lk, lv in (qm.get("layers") or {}).items():
            if isinstance(lv, dict):
                qcfg[lk[len(prefix):] if prefix and lk.startswith(prefix) else lk] = lv
        if qcfg:
            _dbg(f"quantization metadata: {len(qcfg)} layer(s) declared in header")
    except Exception as e:
        _dbg(f"_quantization_metadata unreadable: {e}")
    with safe_open(path, framework="pt", device="cpu") as f:
        for k, _ in entries:
            kk = k[len(prefix):]
            if kk.endswith(".comfy_quant"):  # blob JSON: format + convrot eventuels
                try:
                    qcfg[kk[:-len(".comfy_quant")]] = json.loads(
                        bytes(f.get_tensor(k).tolist()).decode("utf-8"))
                except Exception as e:
                    _dbg(f"comfy_quant blob unreadable {k}: {e}")
                continue
            raw[kk] = f.get_tensor(k)
    # The dequantization work (fp32 cast + scales + un-rotation) is MEMORY BANDWIDTH
    # bound on the CPU (measured on crispz-krea2: ~9 min on a 12.9B INT8): so it runs on
    # the GPU when there is one, tensor by tensor (a few hundred MB of VRAM at most), with
    # the bf16 coming back to RAM. config convert_device: auto (the default) | cpu.
    dev = "cpu"
    try:
        if (torch.cuda.is_available()
                and str(CONFIG.get("convert_device", "auto")).lower() != "cpu"):
            dev = "cuda"
    except Exception:
        pass
    _had = {}                                # a Hadamard cache per group size
    sd = {}
    n_dq = n_rot = n_pre = 0
    for k in list(raw.keys()):
        if (k.endswith((".weight_scale", ".scale_weight", ".scale_input", ".input_scale"))
                or k.endswith("scaled_fp8")):
            continue                         # consommees via lookup / jetees (scale_input
                                             # = an ACTIVATION scale, not a weight one)
        t = raw.pop(k)
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                       torch.int8, torch.uint8):
            s = None
            for cand in (k + "_scale",       # X.weight -> X.weight_scale (ComfyUI)
                         (k[:-len(".weight")] + ".scale_weight")
                         if k.endswith(".weight") else None):
                if cand and cand in raw:
                    s = raw[cand]
                    break
            qdt = t.dtype
            t = t.to(dev).to(torch.float32)
            cfg0 = qcfg.get(k[:-len(".weight")]) if k.endswith(".weight") else None
            if s is not None and _stored_at_scale(t, s, qdt, cfg0):
                s = None                     # already at scale: see _stored_at_scale
                n_pre += 1
            if s is not None:                # scalaire ou [out,1] -> broadcast
                t = t * s.to(dev).to(torch.float32)
            # ConvRot (comfy-quants int8_tensorwise): the stored weights were rotated
            # W_rot = (W.view(out, in/g, g) @ H.T).reshape(...) BEFORE quantization ->
            # reconstruction = multiply by H again (orthonormal, symmetric) per group.
            cfg = qcfg.get(k[:-len(".weight")]) if k.endswith(".weight") else None
            if cfg and cfg.get("convrot"):
                g = int(cfg.get("convrot_groupsize", 256) or 256)
                if t.dim() == 2 and g > 1 and t.shape[1] % g == 0:
                    if g not in _had:
                        _had[g] = _hadamard_ortho(g).to(dev)
                    t = (t.view(t.shape[0], -1, g) @ _had[g]).reshape(t.shape[0], -1)
                    n_rot += 1
            t = t.to(DTYPE).cpu()
            n_dq += 1
        elif t.is_floating_point() and t.dtype != DTYPE:
            t = t.to(DTYPE)
        sd[k] = t
    raw.clear()
    if dev != "cpu":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    _log((f"dequantized {n_dq} tensors on {dev}" if n_dq
          else "read (already bf16/fp16, no dequant)")
         + f" ({len(sd)} kept"
         + (f", {n_rot} un-rotated (ConvRot)" if n_rot else "")
         + (f", {n_pre} already stored at scale: weight_scale NOT applied" if n_pre else "")
         + f") to bf16 in {time.time() - t0:.1f}s")
    return sd


# Architecture expected in .gguf files. A diffusion GGUF declares its architecture in
# 'general.architecture': 'flux' (city96 FLUX.1 dev/schnell/krea), 'qwen_image',
# 'krea2' (Krea 2 = its OWN architecture, which requires ComfyUI + its own encoder/VAE),
# 'llama'/'gemma3'... for LLMs. Only 'qwen_image' is loaded here.
GGUF_ARCH = str(CONFIG.get("gguf_arch") or "qwen_image").strip().lower()

_GGUF_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _gguf_skip(f, t):
    """Skips past a GGUF value in the stream without reading it (strings and arrays
    included)."""
    import struct
    if t == 8:                                   # string
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        return
    if t == 9:                                   # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et in _GGUF_FIXED:
            f.seek(struct.calcsize(_GGUF_FIXED[et]) * n, 1)
        else:
            for _ in range(n):
                _gguf_skip(f, et)
        return
    f.seek(struct.calcsize(_GGUF_FIXED[t]), 1)


def _gguf_arch(path, max_kv=64):
    """The 'general.architecture' of a .gguf -- reads the header only (a few KB), never
    the weights. Returns 'qwen_image' / 'flux' / 'krea2' / 'llama'... or None when
    unreadable (in that case nothing is filtered: better to try than to discard a valid
    model)."""
    import struct
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            f.seek(4 + 8, 1)                     # version (u32) + tensor_count (u64)
            nkv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(min(nkv, max_kv)):
                kl = struct.unpack("<Q", f.read(8))[0]
                if kl > 4096:                    # en-tete incoherent -> on abandonne
                    return None
                key = f.read(kl).decode("utf-8", "replace")
                t = struct.unpack("<I", f.read(4))[0]
                if key == "general.architecture" and t == 8:
                    n = struct.unpack("<Q", f.read(8))[0]
                    return f.read(n).decode("utf-8", "replace").strip().lower()
                _gguf_skip(f, t)
    except Exception as e:
        _dbg(f"gguf header read failed {path}: {e}")
    return None


# Tensor prefixes of the ORIGINAL Qwen layout (the one diffusers' GGUF loader knows how
# to map -- QuantStack/city96 GGUFs). Some Civitai GGUFs are converted with a compact
# renamed scheme (blocks.N.attn.wq, txtmlp, tproj... -- a stable-diffusion.cpp-style tool):
# the declared architecture really is 'qwen_image' but NO key matches -> every weight stays
# on the 'meta' device and the .to(device) blows up with "Cannot copy out of meta tensor".
# That case is detected from the header so it can be refused cleanly.
_GGUF_OK_PREFIXES = ("transformer_blocks.", "img_in", "txt_in", "time_text_embed",
                     "norm_out", "proj_out")

# The signature that POSITIVELY identifies a Qwen-Image transformer, as opposed to the
# other diffusers DiTs. The prefixes above are too loose for that: 'transformer_blocks.',
# 'time_text_embed', 'norm_out' and 'proj_out' exist in FLUX TOO (which names its inputs
# x_embedder/context_embedder). These three keys do not:
_QWEN_GGUF_SIGNATURE = ("img_in.weight", "txt_in.weight", "txt_norm.weight")


def _gguf_layout(path):
    """Layout state of a .gguf's tensors, read from the header (gguf mmap):
      'qwen'    -> the Qwen-Image signature is there: that is PROOF, far more reliable than
                   the declared 'general.architecture' (conversion tools stamp anything --
                   'wan' has been seen on perfectly valid Qwen-Image files);
      'foreign' -> readable names, but no known diffusers marker (a
                   stable-diffusion.cpp-style conversion: blocks.N.attn.wq, txtmlp,
                   tproj...);
      'unknown' -> an unreadable header, or a diffusers layout without the Qwen signature:
                   nothing is decided here, the declared architecture stays the judge.
"""
    try:
        from gguf import GGUFReader
        names = [t.name for t in GGUFReader(path).tensors]
        if not names:
            return "unknown"
        if all(any(n == sig for n in names) for sig in _QWEN_GGUF_SIGNATURE):
            return "qwen"
        if any(n.startswith(_GGUF_OK_PREFIXES) for n in names):
            return "unknown"
        return "foreign"
    except Exception as e:
        _dbg(f"gguf layout check failed {path}: {e}")
        return "unknown"


def _gguf_layout_unsupported(path):
    """Returns a reason (str) when the .gguf does NOT use the Qwen tensor layout diffusers
    expects, otherwise None. Header read only."""
    if _gguf_layout(path) != "foreign":
        return None
    return ("GGUF with a non-standard tensor layout (e.g. stable-diffusion.cpp "
            "conversion); diffusers cannot map it — use a QuantStack/city96-style "
            "GGUF or the BF16/FP16 .safetensors build")


def _checkpoint_dirs():
    """Folders to scan for single-file checkpoints: the main one + the extra one (when
    set), with no duplicate path."""
    dirs = [CHECKPOINTS_DIR]
    if CHECKPOINTS_EXTRA_DIR and CHECKPOINTS_EXTRA_DIR not in dirs:
        dirs.append(CHECKPOINTS_EXTRA_DIR)
    return dirs


def list_checkpoints():
    """Single-file Qwen-Image models (.safetensors / .gguf) from the checkpoints folders
    (main + extra, merged into a single list). ComfyUI 'scaled' FP8/INT8 are accepted (the
    dequant loader, see _safetensors_dequant); only these stay discarded, with their reason:
    stray LoRAs, SVDQuant/Nunchaku INT4, GGUFs of another architecture or in the sd.cpp
    layout. On a duplicate file name, the main folder wins.
"""
    out = []
    seen = set()
    for d in _checkpoint_dirs():
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f in seen:
                continue
            if not f.lower().endswith((".safetensors", ".ckpt", ".pt", ".sft", ".gguf")):
                continue
            if f.lower().endswith(".safetensors"):
                reason = _safetensors_unsupported(os.path.join(d, f))
                if reason:
                    _log(f"checkpoint skipped ({reason}): {f}")
                    continue
            if f.lower().endswith(".gguf"):
                fp = os.path.join(d, f)
                lay, a = _gguf_layout(fp), _gguf_arch(fp)
                # The LAYOUT beats the declared architecture: tensor names are proof,
                # the 'general.architecture' KV a mere label -- and conversion tools stamp
                # it wrong (Qwen-Image files published as 'wan').
                if lay == "foreign":
                    _log(f"checkpoint skipped ({_gguf_layout_unsupported(fp)}): {f}")
                    continue
                if lay == "qwen":
                    if a and a != GGUF_ARCH:
                        _log(f"GGUF declares architecture '{a}' but its tensors ARE a "
                             f"Qwen-Image transformer (mislabelled by the conversion "
                             f"tool) -> loaded anyway: {f}")
                # layout indetermine -> l'archi declaree reste le juge.
                elif a and a != GGUF_ARCH:
                    _log(f"checkpoint skipped (GGUF architecture '{a}', this build only "
                         f"loads '{GGUF_ARCH}'; that model needs its own pipeline and "
                         f"text encoder/VAE): {f}")
                    continue
            seen.add(f)
            out.append(f)
    return sorted(out)


def resolve_checkpoint(name):
    """Absolute path of a single-file checkpoint from its file name, looked up in the
    checkpoints folders (main then extra). Returns name as is when it is already absolute;
    falls back to the main folder when not found."""
    if not name or os.path.isabs(name):
        return name
    for d in _checkpoint_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(CHECKPOINTS_DIR, name)


def list_loras():
    """LoRAs (.safetensors / .ckpt / .pt) from the loras folder, RECURSIVELY (subfolders
    included). Returns paths RELATIVE to LORAS_DIR with '/' (e.g.
    'subfolder/my_lora.safetensors') -> set_loras / resolve resolve them through
    os.path.join(LORAS_DIR, name)."""
    exts = (".safetensors", ".ckpt", ".pt")
    out, seen = [], set()
    for d in _lora_dirs():          # main then extras: same name -> the main one wins
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(exts):
                    rel = os.path.relpath(os.path.join(root, f), d).replace(os.sep, "/")
                    if rel.lower() not in seen:
                        seen.add(rel.lower())
                        out.append(rel)
    return sorted(out)


def set_checkpoints_dir(path):
    global CHECKPOINTS_DIR
    if path:
        CHECKPOINTS_DIR = path


def set_checkpoints_extra_dir(path):
    """Sets (or clears with '' / None) the additional checkpoints folder."""
    global CHECKPOINTS_EXTRA_DIR
    CHECKPOINTS_EXTRA_DIR = (path or "").strip()


def set_loras_dir(path):
    global LORAS_DIR
    if path:
        LORAS_DIR = path


def set_loras_extra_dirs(spec):
    """Sets (or clears with '' / [] / None) the extra LoRA folders.
    spec = a list or an 'a;b' string."""
    global LORAS_EXTRA_DIRS
    LORAS_EXTRA_DIRS = _split_dirs(spec)


def _read_safetensors_metadata(path):
    """Reads the JSON header (__metadata__) of a .safetensors WITHOUT loading the weights."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
    return (json.loads(header.decode("utf-8")) or {}).get("__metadata__", {}) or {}


def lora_keywords(path):
    """Extracts a LoRA's keywords / trigger words from its metadata: explicit trigger
    fields + the top training tags (ss_tag_frequency)."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        meta = _read_safetensors_metadata(path)
    except Exception as e:
        _dbg(f"lora metadata read failed: {e}")
        return ""
    words = []
    for k in ("ss_trigger_words", "modelspec.trigger_phrase", "trigger_words",
              "activation text", "ss_activation_text"):
        v = meta.get(k)
        if v:
            words.append(v if isinstance(v, str) else ", ".join(map(str, v)))
    tf = meta.get("ss_tag_frequency")
    if tf:
        try:
            d = json.loads(tf) if isinstance(tf, str) else tf
            counts = {}
            for ds in d.values():
                for tag, c in ds.items():
                    counts[tag] = counts.get(tag, 0) + int(c)
            words.extend(sorted(counts, key=counts.get, reverse=True)[:15])
        except Exception:
            pass
    seen, out = set(), []
    for w in words:
        for part in str(w).split(","):
            part = part.strip()
            if part and part.lower() not in seen:
                seen.add(part.lower())
                out.append(part)
    return ", ".join(out)


def set_loras(slots):
    """Sets the active LoRAs. slots = a list of (name_or_None, weight). Resolves the
    names to paths, ignores the Nones.

    Does NOT reload the model: the LoRAs are hot-swapped on the transformer already in
    VRAM (_apply_loras, called by _ensure_base on the next run).
"""
    global LORAS
    new = []
    for name, weight in slots:
        if name and name not in ("None", "none", ""):
            new.append((resolve_lora_path(name), float(weight)))
    if new != LORAS:
        LORAS = new
        _log("LoRAs -> " + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in new) or "(none)")
             + " -> applied on next run (hot-swap, no model reload)")


def set_edit_loras(slots):
    """Sets the EDIT LoRAs (the omni pipe). slots = a list of (name_or_None, weight); a
    name is a cz_edit_loras preset (downloaded on demand), an absolute path, or a file
    relative to LORAS_DIR. Ignores the Nones. Applied hot on the next generate_omni
    (_apply_edit_loras). Raises when a preset cannot be downloaded."""
    global EDIT_LORAS
    import cz_edit_loras
    new = []
    for name, weight in slots:
        if not name or name in ("None", "none", ""):
            continue
        name = cz_edit_loras.strip_label(name) if isinstance(name, str) else name
        if cz_edit_loras.spec(name) is not None:
            p = cz_edit_loras.resolve(name)
        else:
            p = resolve_lora_path(name)
        new.append((p, float(weight)))
    if new != EDIT_LORAS:
        EDIT_LORAS = new
        _log("edit LoRAs -> " + (", ".join(f"{os.path.basename(p)}@{w}" for p, w in new)
                                or "(none)") + " -> applied on next edit (hot-swap)")


def edit_speed_choices():
    """Libelles du dropdown 'Edit speed' (Off, Auto, Lightning N steps...)."""
    import cz_edit_loras
    return ["Off"] + cz_edit_loras.speed_names()


def set_edit_speed(name):
    """Fast edit mode. 'Off'/None -> steps/guidance from Settings, no speed LoRA.
    'Auto' -> the edit model's model_profiles profile (Rapid-AIO / a Lightning merge:
    already distilled, 4-8 steps, CFG off), with no LoRA. 'Lightning N steps' -> the
    Lightning LoRA (2509 or 2511 depending on the edit model; looked up in the LoRA
    folders, downloaded otherwise) + N steps + guidance 1.0. Returns the applied dict."""
    global EDIT_SPEED
    import cz_edit_loras
    name = (name or "Off").strip()
    if name.lower() in ("off", "none", ""):
        if EDIT_SPEED is not None:
            _log("edit speed -> off (Settings steps/guidance)")
        EDIT_SPEED = None
        return None
    if name.lower().startswith("auto"):
        from cz_core import profile_for_model
        steps, guidance = profile_for_model(os.path.basename(OMNI_MODEL or ""))
        EDIT_SPEED = {"name": name, "steps": int(steps), "guidance": float(guidance),
                      "path": None}
    else:
        EDIT_SPEED = cz_edit_loras.resolve_speed(name, OMNI_MODEL)
    _log(f"edit speed -> {EDIT_SPEED['name']}: {EDIT_SPEED['steps']} steps, "
         f"cfg {EDIT_SPEED['guidance']:g}"
         + (f", LoRA {os.path.basename(EDIT_SPEED['path'])}" if EDIT_SPEED.get("path") else ""))
    return EDIT_SPEED


def set_edit_loras_enabled(on):
    """The 'Edit LoRAs' checkbox: ON = the EDIT_LORAS set is applied on the omni pipe,
    OFF = removed (the set stays remembered). No reload."""
    global EDIT_LORAS_ENABLED
    on = bool(on)
    if on != EDIT_LORAS_ENABLED:
        EDIT_LORAS_ENABLED = on
        _log(f"edit LoRAs {'enabled' if on else 'disabled'}")


def set_omni_model(repo):
    """Sets the Omni/Edit model (an HF repo or a folder). Invalidates the omni pipe."""
    global OMNI_MODEL, _APPLIED_EDIT_LORAS
    repo = (repo or "").strip()
    if repo != OMNI_MODEL:
        OMNI_MODEL = repo
        _DERIVED.pop("omni", None)
        _APPLIED_EDIT_LORAS = []
        _log(f"Omni model -> {repo or '(none)'}")


def list_edit_models():
    """Local EDIT models (full paths) in the checkpoints folders: single-file files
    (.gguf / .safetensors) whose name contains 'edit', 'aio' or 'rapid' (Qwen-Image-Edit
    2509/2511, Rapid-AIO), stray LoRAs and unloadable formats excluded.
    Feeds the 'Omni model' dropdown (free text is still possible)."""
    out, seen = [], set()
    for d in _checkpoint_dirs():
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d), key=str.lower):
            low = f.lower()
            if not low.endswith((".gguf", ".safetensors")) or f in seen:
                continue
            if not any(k in low for k in ("edit", "aio", "rapid")):
                continue
            # a text encoder (Qwen2.5-VL) / a VAE filed next to them: not
            # transformers
            if any(k in low for k in ("qwen25vl", "qwen2.5", "qwen2_5", "text_encoder",
                                      "textencoder", "vae", "clip")):
                continue
            p = os.path.join(d, f)
            if low.endswith(".safetensors") and _safetensors_unsupported(p):
                continue           # LoRA Lightning d'edition rangee la, SVDQuant...
            seen.add(f)
            out.append(p)
    return out


def check_omni_available():
    """The Edit tab = Qwen-Image-Edit (an instruction-based edit model). Checks that the
    configured edit repo exists on Hugging Face (public API)."""
    import urllib.request
    repo = (OMNI_MODEL or DEFAULT_OMNI_REPO).strip()
    looks_local = (repo.lower().endswith((".gguf", ".safetensors")) or os.path.isdir(repo)
                   or os.path.isabs(repo))
    if looks_local:
        # A LOCAL file/folder: nothing to check on the Hub. Only the rest of the pipe
        # (text encoder, VAE) comes from the base repo on first use.
        if os.path.exists(repo):
            base_edit = (os.environ.get("QWEN_EDIT_BASE") or CONFIG.get("zimage_omni_base")
                         or DEFAULT_OMNI_REPO).strip()
            kind = "GGUF" if _is_gguf_path(repo) else "single-file"
            rev = "2511" if "2511" in repo else "2509"
            return (f"**Edit model ready (local {kind}, {rev}):** `{repo}` "
                    f"({os.path.getsize(repo) / 1024**3:.1f} GB). Text encoder / VAE come "
                    f"from `{base_edit}` on first use.")
        return f"Edit model `{repo}` **not found on disk**. Fix config.txt `zimage_omni_model`."
    try:
        req = urllib.request.Request("https://huggingface.co/api/models/" + repo,
                                     headers={"User-Agent": "crispz-qwen-edit"})
        with urllib.request.urlopen(req, timeout=8) as r:
            if r.status == 200:
                return (f"**Edit model ready:** `{repo}`. The Edit tab edits an input image "
                        "from an instruction prompt. Change it in config.txt "
                        "`zimage_omni_model` (or the Models tab).")
    except Exception:
        pass
    return (f"Edit model `{repo}` not reachable (network/HF). It downloads on first use of "
            "the Edit tab. Override via config.txt `zimage_omni_model`.")


def set_offload_mode(mode):
    """Changes the CPU offload mode. Invalidates the pipe (the hooks are set at load
    time). An unknown value -> 'auto' (never 'none': the fallback must be the SAFE mode)."""
    global OFFLOAD_MODE, _AUTO_OFFLOAD
    mode = str(mode or "").strip().lower()
    mode = mode if mode in OFFLOAD_CHOICES else "auto"
    if mode != OFFLOAD_MODE:
        OFFLOAD_MODE = mode
        _AUTO_OFFLOAD = ""   # 'auto' runs the VRAM test again on the next load
        free_vram()
        _log(f"offload -> {OFFLOAD_MODE}: pipeline invalidated -> will reload")


# ---- Offload 'auto': a VRAM test at load time + a runtime safety net (cz_hw) ----

def _hw_profile_path():
    """Profile of the VRAM test's verdicts (JSON), next to the other caches."""
    return os.path.join(HERE, "cache", "hw_profile.json")


def _model_footprint_gb():
    """VRAM footprint (GB) of the whole pipeline under 'none' offload (weights in VRAM,
    activations excluded). Qwen-Image: a bf16 transformer ~40 GB + the encoder ~17 GB
    -> ~57 GB, so 'auto' resolves 'model' on every consumer card (a ~11-13 GB GGUF
    transformer is forced to 'model' anyway, and the encoder is evicted once the prompt is
    encoded). Overridable through config 'model_footprint_gb' (e.g. a big pro card + a light
    single-file transformer)."""
    try:
        v = float(CONFIG.get("model_footprint_gb", 0) or 0)
        if v > 0:
            return v
    except Exception:
        pass
    return 57.0


def _resolve_auto(retest=False):
    """The concrete mode for 'auto' (memoised for the process). The verdict is cached in
    cache/hw_profile.json per (GPU, torch/cuda build, model, dtype): the test only costs one
    mem_get_info per combination, then a JSON read."""
    global _AUTO_OFFLOAD
    if _AUTO_OFFLOAD and not retest:
        return _AUTO_OFFLOAD
    mode, why = cz_hw.resolve(
        "auto", footprint_gb=_model_footprint_gb(),
        model_id=(ZIMAGE_TRANSFORMER or BASE_REPO), dtype="bf16",
        profile_path=_hw_profile_path(), retest=retest)
    _AUTO_OFFLOAD = mode
    _log(f"offload auto -> {mode} ({why})")
    return mode


def offload_status():
    """Status line for the UI: the requested mode + the 'auto' resolution when relevant."""
    if OFFLOAD_MODE != "auto":
        return f"offload: {OFFLOAD_MODE} (explicit)"
    if not _AUTO_OFFLOAD:
        return "offload: auto (resolves at the next model load)"
    return f"offload: auto -> {_AUTO_OFFLOAD}"


def retest_offload():
    """The UI's 'Re-test VRAM' button: runs the test again, ignoring the profile (another
    app closed/opened, a driver change...). Invalidates the pipe when the verdict changes."""
    if OFFLOAD_MODE != "auto":
        return f"Offload is '{OFFLOAD_MODE}' (explicit) - select 'auto' to use the VRAM test."
    old = _AUTO_OFFLOAD
    mode = _resolve_auto(retest=True)
    if old and mode != old:
        free_vram()
        return f"auto -> {mode} (was {old}; the pipeline will reload)"
    return f"auto -> {mode}"


def _vram_guard_kwargs():
    """Runtime safety net: a callback_on_step_end that checks AFTER the first denoise step
    in effective mode 'none' that the VRAM is not saturated (the load-time test estimates; a
    third-party process may have arrived since, or the requested resolution exceeds the
    margin). Saturated -> the flag + an interruption of the denoise; the caller switches to
    'model' and replays the job ONCE.
    {} when the guard is pointless (offload already on, no CUDA).
"""
    if DEVICE != "cuda" or _effective_offload() != "none":
        return {}

    def _cb(pipe, i, t, cb_kwargs):
        global _VRAM_DOWNGRADE
        if i == 0 and cz_hw.vram_saturated():
            _VRAM_DOWNGRADE = True
            pipe._interrupt = True
        return cb_kwargs
    return {"callback_on_step_end": _cb}


def _consume_vram_downgrade():
    """When the guard has fired: applies the downgrade to 'model', records it in the
    profile (the next boot starts in 'model' directly) and releases the pipe. True -> the
    caller replays the job once."""
    global _VRAM_DOWNGRADE, _AUTO_OFFLOAD
    if not _VRAM_DOWNGRADE:
        return False
    _VRAM_DOWNGRADE = False
    _log("WARNING: VRAM saturated after the first denoise step in offload 'none' "
         "-> the render would spill to shared RAM (50-100x slower, no error). "
         "Switching to 'model' and retrying the job once.")
    cz_hw.record_downgrade(_hw_profile_path(), ZIMAGE_TRANSFORMER or BASE_REPO,
                           "bf16", "model", "VRAM saturated after denoise step 1")
    if OFFLOAD_MODE == "auto":
        _AUTO_OFFLOAD = "model"
        free_vram()
    else:
        set_offload_mode("model")
    return True


def free_vram():
    """Releases the base pipeline + the derived pipelines and gives the VRAM back
    (step 3: unload on idle or the /unload endpoint). Lazy reload."""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _APPLIED_LORAS
    global _TEXT_ENCODER_ACTIVE, _TEXT_ENCODER_ACTIVE_EDIT
    _BASE_PIPE = None
    _DERIVED = {}
    _LOADED_KEY = None
    _APPLIED_LORAS = []      # plus de pipe -> plus d'adaptateur pose
    _TEXT_ENCODER_ACTIVE = ""       # ... nor of a replacement encoder loaded, neither on
    _TEXT_ENCODER_ACTIVE_EDIT = ""  # the base nor on the edit pipe (_DERIVED["omni"])
    _embed_cache_clear(" (VRAM freed)")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def is_oom(e):
    """True when `e` is a lack of VRAM, in either of its two forms: torch's allocator one
    ("CUDA out of memory. Tried to allocate ...") and a direct CUDA call's one ("CUDA error:
    out of memory"). The second happens once torch's cache has reserved everything: a kernel
    loaded on demand finds nothing left and cannot claim anything back from that cache
    (carried over from crispz-klein 1.36.4)."""
    s = str(e).lower()
    return "out of memory" in s or "alloc_failed" in s


def release_vram(offload=False, why=""):
    """Gives the driver back the VRAM that torch's cache holds in reserve, unloading
    nothing.

    torch only empties its cache when ITS allocator fails; the other consumers fail without
    being able to reclaim it. offload=True also puts back on the CPU the models an
    interrupted call left on the GPU under 'model' offload, for EVERY pipeline loaded with
    its hooks (the base one, and the Omni pipeline when it is loaded separately). On
    crispz-klein, a transformer half-moved by an OOM stayed on the GPU: 10.8 GB stuck, and
    every later render failed until a restart. `why` logs the VRAM state afterwards.
"""
    if offload:
        seen = set()
        for p in [_BASE_PIPE, *_DERIVED.values()]:
            if p is None or id(p) in seen or not getattr(p, "_all_hooks", None):
                continue
            seen.add(id(p))
            try:
                p.maybe_free_model_hooks()   # diffusers: everything on the CPU, hooks put back
            except Exception as e:
                _dbg(f"release_vram: offload failed ({e})")
    gc.collect()
    if DEVICE != "cuda":
        return
    try:
        torch.cuda.empty_cache()
        if why:
            free, total = torch.cuda.mem_get_info()
            _log(f"VRAM released ({why}): {free / 1024 ** 3:.1f} GB free of "
                 f"{total / 1024 ** 3:.1f}, torch holds "
                 f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB "
                 f"(reserved {torch.cuda.memory_reserved() / 1024 ** 3:.1f})")
    except Exception as e:
        _dbg(f"release_vram: {e}")


def _load_lora(pipe, *args, **kwargs):
    """pipe.load_lora_weights with REAL tensors (low_cpu_mem_usage=False).

    A diffusers/peft build that does not know that parameter refuses it with a TypeError:
    the call is then retried without it rather than failing the application (the default
    creates the layers on 'meta' there -- see _sync_adapters).
"""
    try:
        return pipe.load_lora_weights(*args, low_cpu_mem_usage=False, **kwargs)
    except TypeError as e:
        if "low_cpu_mem_usage" not in str(e):
            raise
        _dbg(f"load_lora_weights without low_cpu_mem_usage ({e})")
        return pipe.load_lora_weights(*args, **kwargs)


def _offload_hooks(pipe):
    """Number of 'model' offload hooks diffusers has set on this pipe (0 = none)."""
    return len(getattr(pipe, "_all_hooks", None) or [])


def restore_offload(pipe, why=""):
    """Puts the pipe back in its EFFECTIVE offload state when it has been left on the CPU.

    diffusers REMOVES the offload hooks before applying a LoRA and puts them back after.
    When the load fails in between, nobody puts them back: the pipe stays on the CPU, its
    `_execution_device` becomes cpu, and EVERY later render fails on "Cannot generate a cpu
    tensor from a generator of type cuda" -- until the app is restarted. Caught on 2026-09-23
    with two DoRA LoRAs (crispz-klein 1.36.6). Returns True when the state has been restored.
"""
    if DEVICE != "cuda" or pipe is None:
        return False
    try:
        dev = pipe._execution_device
    except Exception:
        return False
    if str(getattr(dev, "type", dev)) == "cuda":
        return False
    off = _effective_offload()
    try:
        if off == "model":
            pipe.enable_model_cpu_offload()
        elif off == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(DEVICE)
    except Exception as e:
        _log(f"pipeline left on the CPU and NOT restored ({e}): restart crispz-qwen-edit")
        return False
    _log(f"pipeline was left on the CPU{' after ' + why if why else ''} -> offload "
         f"'{off}' restored in place (no reload)")
    return True


def retry_on_oom(what, fn, *args, **kwargs):
    """Calls fn(*args, **kwargs); on a lack of VRAM, gives the VRAM back (torch's cache,
    models left on the GPU) and retries ONCE. A second failure gives the VRAM back again
    before re-raising: the process stays usable for the next render."""
    err = None
    for attempt in (1, 2):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not is_oom(e):
                raise
            # The traceback holds the frames, so their tensors on the GPU: it is
            # dropped BEFORE emptying the cache, or empty_cache reclaims nothing.
            err = e.with_traceback(None)
            err.__context__ = err.__cause__ = None
        if attempt == 1:
            _log(f"{what}: out of VRAM ({str(err).strip().splitlines()[0]}), "
                 f"freeing it and retrying once")
        release_vram(offload=True, why=what)
    raise err


# Beyond this side (px) attention slicing is turned on (whole-image 2K+ -> avoids the
# 32 GB VRAM spill). Below it (1024 tiles, 1024/1536 txt2img) -> slicing OFF = native SDPA
# attention = FAST (like ComfyUI). Tunable through config attention_slice_above.
_SLICE_ABOVE = int(CONFIG.get("attention_slice_above", 1664))

# Guard rail: beyond this side (px), a "whole image" refine (refine_tile=0) is
# auto-tiled (1024 tile). The default = the slicing threshold: beyond it a whole-image pass
# would be sliced (slow: ~120s at 2K) AND risks the VRAM spill (4K -> a crash). Tiling is
# faster AND safe.
_AUTO_TILE_ABOVE = int(CONFIG.get("auto_refine_tile_above", _SLICE_ABOVE))

# Size of the tile this auto-tiling uses. "auto" (the default) = computed by
# _pick_refine_tile; an integer freezes the size (the old behaviour: 1024).
# Measured (RTX 5090, 4096x4096 output, denoise 0.40, overlap 64): the cost per pixel is
# FLAT from 768 to 1024 (1.78 / 1.83 / 1.79 us/px) and only climbs beyond that (2.41 at
# 1536, 3.00 at 2048). So the time follows the TILED AREA (n x tile^2), not the tile size.
# But at 1024 the grid overflows: 960 does not divide 4096 -> the last tile is pulled back
# and overlaps the previous one by 832px instead of 64, that is 1.56x the image's area. At
# 896 the step lands right (1.20x) -> 36.7s instead of 46.9s on the same image, with an
# IDENTICAL number of tiles (25) and seams (8).
# Bounds [768, 1024]: below them tiles and seams multiply and each tile sees less context
# (the render drifts - a blurred background rebuilds differently, checked visually); above
# them the attention becomes superlinear.
_AUTO_TILE_MIN = int(CONFIG.get("auto_refine_tile_min", 768))
_AUTO_TILE_MAX = int(CONFIG.get("auto_refine_tile_max", 1024))
_AUTO_TILE_SIZE = str(CONFIG.get("auto_refine_tile", "auto")).strip().lower()


def _pick_refine_tile(w, h, overlap):
    """The tile that minimises the tiled area needed to cover w x h (= the pass's real
    cost).

    At equal area the LARGEST tile wins: fewer seams and more context per tile. An integer
    in auto_refine_tile short-circuits the computation (a frozen size).
"""
    if _AUTO_TILE_SIZE not in ("auto", "", "0"):
        try:
            return round_to_multiple(int(_AUTO_TILE_SIZE))
        except ValueError:
            _log(f"config auto_refine_tile='{_AUTO_TILE_SIZE}' invalide (attendu 'auto' ou "
                 "un entier) -> calcul automatique")
    lo = max(256, _AUTO_TILE_MIN)
    hi = max(lo, _AUTO_TILE_MAX)
    ov = max(0, int(overlap))
    cands = []
    for t in range(lo, hi + 1, 32):
        step = max(16, t - ov)
        n = len(range(0, max(1, int(w)), step)) * len(range(0, max(1, int(h)), step))
        cands.append((n * t * t, -t, t))       # the smallest area, then the largest tile
    return min(cands)[2]

# Denoise ceiling for the TILED refine. In tiles, each tile is re-diffused with the
# global prompt -> at a high denoise the diffusion rebuilds the subject (the cup, say) IN
# every tile = duplications. So the per-tile denoise is capped (the existing content then
# guides the diffusion, Ultimate SD Upscale style). The "whole image" refine keeps the
# requested denoise (no duplication is possible: a single pass over the whole composition).
# Tunable through config refine_tile_denoise_cap (0 = no cap).
_TILE_DENOISE_CAP = float(CONFIG.get("refine_tile_denoise_cap", 0.40))

# Prompt used for the TILED refine. The global prompt describes the WHOLE composition
# (not the tile) -> handing it to every tile pushes the diffusion to recreate the subject
# (the cup) in tiles that are nothing but background. So an EMPTY prompt is passed by
# default: each tile just refines the local detail. config refine_tile_prompt values:
#   "" (the default) = an empty prompt per tile
#   "global"/"scene" = reuses the scene's prompt (the old behaviour)
#   any other text = a generic prompt applied to every tile (e.g. "high detail, sharp")
_TILE_PROMPT = str(CONFIG.get("refine_tile_prompt", ""))


def _tile_prompt(scene_prompt):
    """The prompt to use per tile according to the config (empty by default,
    anti-duplication)."""
    if _TILE_PROMPT.strip().lower() in ("global", "scene"):
        return scene_prompt or ""
    return _TILE_PROMPT


def _set_slicing(pipe, longest_side):
    """Turns attention slicing on/off according to the largest side to process. Called
    before EVERY diffusion pass (txt2img/refine/tile/inpaint/outpaint/omni)."""
    try:
        if int(longest_side) > _SLICE_ABOVE:
            pipe.enable_attention_slicing()
        else:
            pipe.disable_attention_slicing()
    except Exception:
        pass


def _vram_str():
    """PyTorch's peak reserved VRAM / the total (to spot saturation -> a spill into
    Windows' shared RAM = extreme slowness, and TDR/'CUDA unknown error'). Does NOT see the
    other processes' VRAM (ComfyUI, etc.) -> use nvidia-smi for the real total."""
    if DEVICE != "cuda":
        return ""
    try:
        resv = torch.cuda.memory_reserved() / 1024**3
        tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f" | VRAM {resv:.1f}/{tot:.0f} Go"
    except Exception:
        return ""


# ----------------------------------------------------------------------------
# Qwen-Image (diffusers, BF16): a "base" txt2img pipeline that owns the components, with
# img2img / inpaint derived through from_pipe (shared weights, no duplicate VRAM).
# ----------------------------------------------------------------------------
def _is_gguf_path(p):
    return bool(p) and str(p).lower().endswith(".gguf")


def _effective_offload(tpath=None):
    """The offload REALLY applied. A quantized GGUF transformer does not move onto the GPU
    through .to(cuda) nor in sequential -> only enable_model_cpu_offload places it on the GPU
    during the forward. So 'model' is forced for a GGUF base, whatever the setting says."""
    off = OFFLOAD_MODE
    if off == "auto":
        off = _resolve_auto()   # test VRAM (memoise + profil cache) -> mode concret
    t = ZIMAGE_TRANSFORMER if tpath is None else tpath
    if DEVICE == "cuda" and _is_gguf_path(t) and off != "model":
        off = "model"
    return off


def _load_transformer(path=None, base=None):
    """Loads ONLY the current transformer (without the rest of the pipeline):
      - a quantized GGUF -> from_single_file + GGUFQuantizationConfig (architecture = the
        base repo's)
      - a single-file .safetensors -> from_single_file
      - an HF repo / diffusers folder -> the 'transformer' subfolder
      - no override -> the base repo's transformer
    Used both on a full load AND for the hot swap (_swap_transformer).
    path/base: by default the BASE's transformer (ZIMAGE_TRANSFORMER / BASE_REPO); the EDIT
    pipe passes its own file (GGUF, FP8 Rapid-AIO...) + its edit repo (zimage_omni_base) for
    the architecture.
"""
    from diffusers import QwenImageTransformer2DModel
    path = ZIMAGE_TRANSFORMER if path is None else path
    base = BASE_REPO if base is None else base
    if path:
        if _is_single_file(path):
            # Guard: a file that cannot be loaded (a stray LoRA, an FP8, a quantized
            # one) must fail with an actionable message, not go looking for a default
            # config on the Hub. (No effect on .gguf files: an unreadable header -> None.)
            bad = _safetensors_unsupported(path)
            if bad:
                raise RuntimeError(f"{os.path.basename(path)}: {bad}.")
            if _is_gguf_path(path):
                # A Qwen GGUF transformer (quantized) -> fits in VRAM (~11 GB in Q4)
                # and stays fast. The VAE + text encoder come from the base repo (cached).
                lay = _gguf_layout_unsupported(path)
                if lay:
                    raise RuntimeError(
                        f"{os.path.basename(path)}: {lay}.")
                from diffusers import GGUFQuantizationConfig
                _log(f"loading Qwen transformer (GGUF, quantized): {path} ...")
                # config/subfolder = the transformer's architecture from the base repo
                # (cached), otherwise from_single_file does not know the structure and
                # tries a default repo.
                return _load_monitor(
                    f"transformer {os.path.basename(path)} (GGUF)",
                    lambda: QwenImageTransformer2DModel.from_single_file(
                        path,
                        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
                        config=base, subfolder="transformer",
                        torch_dtype=DTYPE))
            dq = _safetensors_dequant(path)
            if dq:
                # ComfyUI 'scaled' FP8/INT8 (light Civitai builds) -> dequantized in
                # RAM then the dict is loaded (diffusers key conversion included).
                # Already dequantized once? -> re-read the bf16 from the disk cache: a
                # normal single-file load (seconds) instead of converting the whole file
                # again (minutes on a HDD).
                cached = _dequant_cache_path(path)
                if cached and os.path.isfile(cached):
                    _log(f"loading Qwen transformer ({dq} -> bf16, from dequant "
                         f"cache): {os.path.basename(cached)}")
                    try:
                        os.utime(cached, None)       # marks the use for the LRU
                    except OSError:
                        pass
                    return _load_monitor(
                        f"transformer {os.path.basename(path)} (cached bf16)",
                        lambda: QwenImageTransformer2DModel.from_single_file(
                            cached, config=base, subfolder="transformer",
                            torch_dtype=DTYPE))
                _log(f"loading Qwen transformer (single-file, {dq} ComfyUI -> "
                     f"dequantized to bf16): {path} ...")
                sd = _load_dequant_state_dict(path)
                _dequant_cache_store(path, sd)
                return _load_monitor(
                    f"transformer {os.path.basename(path)} ({dq})",
                    lambda: QwenImageTransformer2DModel.from_single_file(
                        sd, config=base, subfolder="transformer",
                        torch_dtype=DTYPE))
            # A single-file Qwen checkpoint (.safetensors bf16/fp16) -> a transformer
            # override. config/subfolder = the transformer's architecture from the base repo
            # (already cached), as for the GGUF: without it, from_single_file does not know
            # the structure and goes looking for a default repo -> a failure in offline mode
            # (HF_HUB_OFFLINE=1).
            if _safetensors_comfy_prefixed(path):
                # ComfyUI layout WITHOUT quantization: diffusers does not convert the
                # Qwen keys (identity mapping), and handing the path over directly would
                # leave the model on 'meta'. So we read and strip the prefix ourselves --
                # the same RAM cost, since from_single_file loads the whole checkpoint
                # anyway. No disk cache here: there is nothing dequantized to memorise, it
                # would be a bf16 -> bf16 copy.
                _log(f"loading Qwen transformer (single-file, ComfyUI layout -> "
                     f"diffusers): {path} ...")
                sd = _load_dequant_state_dict(path)
                return _load_monitor(
                    f"transformer {os.path.basename(path)}",
                    lambda: QwenImageTransformer2DModel.from_single_file(
                        sd, config=base, subfolder="transformer",
                        torch_dtype=DTYPE))
            _log(f"loading Qwen transformer (single-file): {path} ...")
            return _load_monitor(
                f"transformer {os.path.basename(path)}",
                lambda: QwenImageTransformer2DModel.from_single_file(
                    path, config=base, subfolder="transformer",
                    torch_dtype=DTYPE))
        # An HF repo / diffusers folder -> load the 'transformer' subfolder.
        _log(f"loading Qwen transformer (repo subfolder): {path} ...")
        return _load_monitor(
            f"transformer {path}",
            lambda: QwenImageTransformer2DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=DTYPE))
    _log(f"loading Qwen transformer (base repo): {base} ...")
    return _load_monitor(
        f"transformer {base}",
        lambda: QwenImageTransformer2DModel.from_pretrained(
            base, subfolder="transformer", torch_dtype=DTYPE))


def _lora_names(loras):
    return [f"cz_lora_{i}" for i in range(len(loras))]


def _clear_loras(pipe):
    """Removes EVERY LoRA adapter from the pipe to start from a clean state.

    unload_lora_weights() alone leaves, depending on the diffusers/peft versions, a residual
    peft_config on the transformer -> the next load warns ('Already found a peft_config')
    and, since the same adapter names are reused (cz_lora_i), the old adapter can stay in
    place (the wrong LoRA applied). So the remaining adapters are deleted explicitly by name
    after the unload.
"""
    try:
        pipe.unload_lora_weights()
    except Exception as e:
        _dbg(f"unload_lora_weights: {e}")
    try:
        listed = pipe.get_list_adapters() or {}
        names = sorted({n for lst in listed.values() for n in (lst or [])})
        if names:
            pipe.delete_adapters(names)
            _dbg(f"cleared leftover LoRA adapters: {names}")
    except Exception as e:
        _dbg(f"delete_adapters: {e}")


def _sync_adapters(pipe, wanted, applied, force=False, tag="LoRA"):
    """Synchronises a pipe's PEFT adapters with the `wanted` set, WITHOUT reloading the
    model. `applied` = the set really applied on this pipe (a list of (path, weight)).

    The transformer stays in VRAM; only the PEFT adapters move:
      - same files, different weights -> set_adapters (immediate)
      - a different LoRA set          -> unload_lora_weights + reloading the LoRAs (~1s)
    The derived pipes (from_pipe) share that transformer -> they follow automatically.
    Returns (ok, applied): ok=False on a failure (the caller decides: a full reload for the
    base, a hard error for editing), applied = the new set applied ([] on a failure).
"""
    # low_cpu_mem_usage=False on EVERY load (see _load_lora): diffusers' default
    # creates the adapter's layers on 'meta' then copies the weights into them. A DoRA whose
    # 'dora_scale' keys diffusers filters out then leaves parameters without data, and the
    # first move raises "Cannot copy out of meta tensor". With real tensors, a missing key
    # keeps its initial value.
    wanted = list(wanted)
    if not force and applied == wanted:
        return True, applied
    had_hooks = _offload_hooks(pipe)
    old_paths = [p for p, _ in applied]
    new_paths = [p for p, _ in wanted]
    try:
        if not force and old_paths and old_paths == new_paths:
            # Only the weights change -> an instant re-weighting.
            pipe.set_adapters(_lora_names(wanted), [float(w) for _, w in wanted])
            _log(f"{tag} weights updated in place (no reload): "
                 + ", ".join(f"{os.path.basename(p)}@{w}" for p, w in wanted))
            return True, wanted
        if old_paths or force:
            _clear_loras(pipe)
        names, weights = [], []
        for i, (p, w) in enumerate(wanted):
            if os.path.isfile(p):
                an = f"cz_lora_{i}"
                _log(f"applying {tag}: {os.path.basename(p)} (weight {w})")
                # Pass the folder + weight_name (not the full path): otherwise
                # diffusers in offline mode (HF_HUB_OFFLINE) refuses with "must specify a
                # weight_name". Works online too, and with a direct local file.
                # From the 2nd adapter on, peft warns "Already found a peft_config":
                # stacking several LoRAs is precisely the point, so that message is
                # silenced.
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Already found a `peft_config`.*")
                    _load_lora(pipe, os.path.dirname(p) or ".",
                               weight_name=os.path.basename(p), adapter_name=an)
                names.append(an)
                weights.append(float(w))
            else:
                _log(f"{tag} file not found, ignored: {p}")
        if names:
            pipe.set_adapters(names, weights)
        if not force:
            _log(f"{tag}s hot-swapped (no model reload)")
        return True, wanted
    except Exception as e:
        _log(f"{tag} hot-swap failed ({e})")
        # diffusers removed the offload hooks before loading and did not get to put
        # them back: without this, the pipe stays on the CPU and EVERY later render fails,
        # including the ones that have nothing to do with this LoRA.
        if had_hooks and not _offload_hooks(pipe):
            restore_offload(pipe, f"a failed {tag} load")
        return False, []


def _apply_loras(pipe, force=False):
    """BASE LoRAs (txt2img/img2img): synchronises the pipe with LORAS through
    _sync_adapters. Returns True when applied, False on a failure (the caller falls back to
    a full reload)."""
    global _APPLIED_LORAS
    ok, _APPLIED_LORAS = _sync_adapters(pipe, LORAS, _APPLIED_LORAS, force=force)
    if not ok:
        _log("falling back to a full reload")
    return ok


def _apply_edit_loras(pipe):
    """EDIT LoRAs: synchronises the omni pipe with EDIT_LORAS (or [] when the 'Edit LoRAs'
    box is unticked). A failure is a hard error: the user asked for that preset, and an edit
    WITHOUT it would be a false result."""
    global _APPLIED_EDIT_LORAS
    wanted = list(EDIT_LORAS) if EDIT_LORAS_ENABLED else []
    # The fast mode's Lightning LoRA: stacked AFTER the presets (independent of the
    # 'Edit LoRAs' box, which only concerns the task presets).
    if EDIT_SPEED and EDIT_SPEED.get("path"):
        wanted.append((EDIT_SPEED["path"], 1.0))
    ok, _APPLIED_EDIT_LORAS = _sync_adapters(pipe, wanted, _APPLIED_EDIT_LORAS,
                                             tag="edit LoRA")
    if not ok:
        raise RuntimeError("edit LoRA could not be applied on the edit pipe "
                           "(see log); nothing was generated")


def _swap_transformer(pipe):
    """Replaces ONLY the transformer of the already cached pipeline: the VAE, the text
    encoder, the tokenizer and the scheduler stay in VRAM (they are most of the load time).
    Valid only with an identical base repo + EFFECTIVE offload.

    Returns True when the swap succeeded, False -> the caller does a full reload.
"""
    global _APPLIED_LORAS, _DERIVED
    t0 = time.time()
    old_t = _LOADED_KEY[1] if _LOADED_KEY else None
    # Switching to/from a GGUF changes the EFFECTIVE offload (a GGUF forces 'model')
    # -> the accelerate hooks and the placement differ: no tinkering, we reload.
    if _effective_offload(old_t) != _effective_offload(ZIMAGE_TRANSFORMER):
        _log("transformer swap skipped (GGUF changes the effective offload) -> full reload")
        return False
    try:
        _log(f"switching Qwen transformer -> {ZIMAGE_TRANSFORMER or BASE_REPO} "
             "(keeping VAE + text encoder in VRAM)")
        new_t = _load_transformer()
        old = getattr(pipe, "transformer", None)
        off = _effective_offload()
        # Offload: the accelerate hooks are set on the components. They have to be
        # removed before the swap, or the new transformer has none and the old one keeps
        # its own.
        if DEVICE == "cuda" and off in ("model", "sequential"):
            try:
                pipe.remove_all_hooks()
            except Exception as e:
                _dbg(f"remove_all_hooks: {e}")
        try:
            pipe.register_modules(transformer=new_t)   # API diffusers (met a jour le config)
        except Exception:
            pipe.transformer = new_t
        # Free the OLD transformer BEFORE putting the new one on the GPU: otherwise
        # old + new + VAE/encoder exceed the VRAM -> a spill into shared RAM that never
        # recovers (measured on a multi-checkpoint XYZ grid on the studio side: 1.7 s/step
        # -> 300-600 s/step, then a crash). The derived pipes (from_pipe) point at the old
        # one too -> purge them first, or `del old` frees nothing (from_pipe is free, it
        # will be rebuilt).
        _DERIVED = {}
        del old
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if DEVICE == "cuda":
            if off == "model":
                pipe.enable_model_cpu_offload()
            elif off == "sequential":
                pipe.enable_sequential_cpu_offload()
            else:
                new_t.to(DEVICE)       # never a GGUF here (offload forced to 'model')
        # The LoRA adapters were applied on the old transformer -> to be reapplied.
        _APPLIED_LORAS = []
        if LORAS:
            _apply_loras(pipe, force=True)
        _log(f"transformer switched in {time.time() - t0:.1f}s "
             "(VAE + text encoder kept, no full reload)")
        return True
    except Exception as e:
        _log(f"transformer hot-swap failed ({e}); falling back to a full reload")
        _APPLIED_LORAS = []
        return False


def _ensure_base():
    """Loads (when needed) the base txt2img pipeline. Handles the single-file/GGUF
    transformer and the offload. Cached by (repo, transformer, offload).

    Two hot swaps avoid a full reload (transformer + VAE + text encoder, tens of seconds):
      - different LoRAs            -> _apply_loras (the PEFT adapters alone)
      - a different transformer, same base repo + offload -> _swap_transformer.
"""
    global _BASE_PIPE, _DERIVED, _LOADED_KEY, _BASE_SCHED_CONFIG, _APPLIED_LORAS
    global _TEXT_ENCODER_ACTIVE
    key = (BASE_REPO, ZIMAGE_TRANSFORMER, OFFLOAD_MODE)
    _dbg(f"_ensure_base key={key} cached={_LOADED_KEY}")
    if _BASE_PIPE is not None and _LOADED_KEY == key:
        # Safety net: a pipe left on the CPU by an earlier failure (LoRA, offload)
        # would make THIS render fail on "Cannot generate a cpu tensor from a generator of
        # type cuda", and every later one too.
        restore_offload(_BASE_PIPE, "an earlier failure")
        if _apply_loras(_BASE_PIPE):
            _dbg("base pipeline: reusing cached (no reload)")
            return _BASE_PIPE
        _dbg("base pipeline: LoRA hot-swap failed -> free + reload")
        free_vram()
    elif _BASE_PIPE is not None:
        # Only the transformer changes (same base repo + same offload)? -> reload the
        # transformer ONLY and keep VAE + text encoder in VRAM.
        if (_LOADED_KEY and _LOADED_KEY[0] == BASE_REPO and _LOADED_KEY[2] == OFFLOAD_MODE
                and _swap_transformer(_BASE_PIPE)):
            _LOADED_KEY = key
            return _BASE_PIPE
        _dbg("base pipeline: key changed -> free + reload")
        free_vram()
    from diffusers import QwenImagePipeline
    t0 = time.time()
    kwargs = {}
    if ZIMAGE_TRANSFORMER:
        kwargs["transformer"] = _load_transformer()
    # A replacement encoder: checked against BASE_REPO's config then loaded with the
    # repo's class. An encoder that does not suit (the repo changed since the choice, an
    # unreadable folder, a failed load) is dropped WITH a log line, and the metadata says so.
    # img2img / inpaint share it through from_pipe.
    _enc, _TEXT_ENCODER_ACTIVE = _pick_text_encoder(BASE_REPO, "base")
    if _enc is not None:
        kwargs["text_encoder"] = _enc
    _off_label = (f"auto->{_resolve_auto()}" if OFFLOAD_MODE == "auto" else OFFLOAD_MODE)
    _log(f"loading Qwen-Image base: {BASE_REPO} (offload={_off_label}, dtype=bf16) ... "
         "first time downloads from HF (~20B, large), then cached")
    pipe = _load_monitor(f"Qwen-Image base {BASE_REPO}",
                         lambda: QwenImagePipeline.from_pretrained(BASE_REPO, torch_dtype=DTYPE,
                                                                   **kwargs))
    # Capture the scheduler's native (flow-matching) config -> the base for building
    # the other samplers (euler/dpm2a/dpmpp2m) without losing shift/flow params.
    try:
        _BASE_SCHED_CONFIG = dict(pipe.scheduler.config)
    except Exception:
        _BASE_SCHED_CONFIG = None
    # Qwen-Image LoRAs (on the base's transformer -> shared by the derived pipes).
    # force=True: a new pipe, no adapter applied -> we (re)apply everything.
    _APPLIED_LORAS = []
    if LORAS:
        _apply_loras(pipe, force=True)
    # Attention slicing: SET PER CALL through _set_slicing (according to the
    # resolution processed), NOT at load time. In tiles/at 1024 -> slicing OFF = native SDPA
    # attention, fast (like ComfyUI). Whole-image 2K+ -> slicing ON to avoid the 32 GB VRAM
    # spill.
    # enable_*_cpu_offload handles the device itself -> do NOT call .to(cuda) then.
    # IMPORTANT: a quantized GGUF transformer does NOT move onto the GPU through .to(cuda)
    # (offload=none) nor in sequential -> it stays on the CPU = ULTRA slow (empty VRAM,
    # ~500s/step). Only enable_model_cpu_offload (accelerate) places it properly on the GPU
    # during the forward. So 'model' is forced for a GGUF base, whatever the UI/config says.
    _off = _effective_offload()
    _base_off = _resolve_auto() if OFFLOAD_MODE == "auto" else OFFLOAD_MODE
    if _off != _base_off:
        _log(f"GGUF base: offload '{_base_off}' forced to '{_off}' (a GGUF does not run "
             f"on the GPU in none/sequential -> otherwise CPU, ~500s/step)")
    if DEVICE == "cuda" and _off == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and _off == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    # VAE tiling/slicing: essential for img2img/upscale. Qwen-Image is big (~20B
    # transformer + text encoder) -> without tiling the VAE can overflow the VRAM (a spill
    # into shared RAM = very slow). Tiling the VAE caps that peak (like ComfyUI's "tiled
    # decode"). The VAE is shared by the derived pipes.
    try:
        pipe.vae.config.force_upcast = False   # the VAE in bf16 (fp32 is slow on Blackwell) -- ALWAYS
    except Exception:
        pass
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception as e:
        _dbg(f"VAE tiling not available: {e}")
    _apply_sampler(pipe)   # applies the chosen sampler (euler by default) to the base pipe
    _BASE_PIPE = pipe
    _DERIVED = {"txt2img": pipe}
    _LOADED_KEY = key
    _log(f"Qwen-Image base ready in {time.time() - t0:.1f}s (sampler={SAMPLER}/{SCHEDULE})")
    return pipe


def get_pipe(kind="img2img"):
    """Returns the requested pipeline. txt2img/img2img/inpaint derive from the base through
    from_pipe (shared weights). Omni needs extra components (SigLIP) -> loaded separately
    from a dedicated Omni model (CONFIG['zimage_omni_model'])."""
    if kind == "omni":
        # Qwen-Image-Edit is a SEPARATE model: do NOT load the base one (txt2img,
        # ~8 min + duplicate RAM/VRAM) just to edit an image.
        if "omni" in _DERIVED:
            _dbg("get_pipe('omni'): reuse derived")
            return _DERIVED["omni"]
        return _load_omni()
    base = _ensure_base()
    # A sampler/schedule change asked for DURING a render was only recorded
    # (see _reapply_sampler_all): this is where it lands, between two
    # generations, with _GPU_LOCK held by the caller.
    _apply_sampler_if_dirty()
    if kind in _DERIVED:
        _dbg(f"get_pipe('{kind}'): reuse derived")
        return _DERIVED[kind]
    from diffusers import QwenImageImg2ImgPipeline, QwenImageInpaintPipeline
    cls = {"img2img": QwenImageImg2ImgPipeline, "inpaint": QwenImageInpaintPipeline}.get(kind)
    if cls is None:
        return base
    _log(f"deriving {kind} pipeline (shared weights, no extra VRAM)")
    # A GGUF transformer is QUANTIZED: it cannot be recast to a dtype (.to(DTYPE)
    # raises "Casting a quantized model is unsupported"). So the bf16 recast is skipped in
    # that case (the compute_dtype is bf16 already). Otherwise (full bf16): a defensive
    # Blackwell recast (some from_pipe calls upcast to float32 -> very slow without fp32
    # tensor cores).
    quantized = bool(ZIMAGE_TRANSFORMER) and ZIMAGE_TRANSFORMER.lower().endswith(".gguf")
    try:
        # A quantized GGUF: torch_dtype=None EXPLICITLY -> otherwise from_pipe puts
        # float32 by default and casts the quantized model -> ValueError "Casting a
        # quantized model".
        p = cls.from_pipe(base, torch_dtype=None) if quantized else cls.from_pipe(base, torch_dtype=DTYPE)
    except TypeError:
        p = cls.from_pipe(base)
    try:
        if not quantized:
            p = p.to(DTYPE)
        p.vae.config.force_upcast = False
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        _log(f"img2img bf16 recast failed ({e})")
    _apply_sampler(p)   # the same sampler as the base (in case from_pipe recreates the scheduler)
    # Speed diagnosis: if the derived pipe is NOT on cuda -> img2img/refine runs on the
    # CPU = ultra slow. So it is forced onto DEVICE in full-VRAM mode (offload handles
    # itself).
    # NB: the EFFECTIVE offload (a GGUF base forces 'model' even when the UI says 'none'):
    # under offload, a transformer "on the CPU" is normal -> a .to(cuda) would break the
    # hooks.
    try:
        tdev = next(p.transformer.parameters()).device
        if DEVICE == "cuda" and _effective_offload() == "none" and tdev.type != "cuda":
            _log(f"{kind} pipeline was on {tdev} -> moving to {DEVICE}")
            p = p.to(DEVICE)
            tdev = next(p.transformer.parameters()).device
        _log(f"{kind} pipeline ready: transformer={tdev}")
    except Exception as e:
        _dbg(f"device check failed: {e}")
    _DERIVED[kind] = p
    return p


def _load_omni():
    """Loads the Qwen-Image-Edit editing pipeline (the Omni/Edit tab). A model SEPARATE
    from the base (default 'Qwen/Qwen-Image-Edit-2509', multi-image). 2509 ->
    QwenImageEditPlus; the base revision -> QwenImageEdit. A separate pipeline (it shares
    nothing with the base)."""
    global _DERIVED, _APPLIED_EDIT_LORAS, _TEXT_ENCODER_ACTIVE_EDIT
    import diffusers
    _APPLIED_EDIT_LORAS = []        # a new pipe: no edit adapter applied on it
    _TEXT_ENCODER_ACTIVE_EDIT = ""  # ... nor a replacement encoder, as long as it is not mounted
    repo = (OMNI_MODEL or os.environ.get("ZIMAGE_OMNI_MODEL")
            or CONFIG.get("zimage_omni_model") or DEFAULT_OMNI_REPO).strip()
    if not repo:
        raise RuntimeError("No Qwen-Image-Edit model set (config.txt 'zimage_omni_model').")
    EditPlus = getattr(diffusers, "QwenImageEditPlusPipeline", None)
    t0 = time.time()
    if _is_single_file(repo):
        # Single-file: a local edit transformer (a quantized GGUF ~13 GB, or a
        # .safetensors FP8/bf16 such as Rapid-AIO = 2511 + Lightning merged) + the REST (the
        # text encoder ~17 GB, the VAE, the processor) taken from the base edit repo
        # (zimage_omni_base, default Qwen-Image-Edit-2509). Makes editing fit in 32 GB of
        # VRAM, without the 40 GB transformer.
        import importlib, json as _json
        from huggingface_hub import hf_hub_download
        base_edit = (os.environ.get("QWEN_EDIT_BASE") or CONFIG.get("zimage_omni_base")
                     or DEFAULT_OMNI_REPO).strip()
        EditCls = EditPlus or diffusers.QwenImageEditPipeline
        _log(f"loading Qwen-Image-Edit transformer (single-file): {repo} + base {base_edit} "
             f"via {EditCls.__name__} (offload={OFFLOAD_MODE}) ...")
        # The transformer from the file (its architecture taken from the base repo, no
        # download of the bf16 transformer): the same loader as the base one (GGUF, 'scaled'
        # FP8/INT8 dequantized + the disk cache, the ComfyUI layout). NB:
        # from_pretrained(base, transformer=tf) would download the repo's 40 GB transformer
        # ANYWAY -> so the pipeline is built component by component. The classes come from
        # model_index.json.
        tf = _load_transformer(repo, base_edit)
        mi = _json.load(open(hf_hub_download(base_edit, "model_index.json"), encoding="utf-8"))
        comps = {"transformer": tf}
        # A replacement encoder: compared against base_edit's config (the repo that
        # supplies THIS pipe's encoder) and substituted into comps['text_encoder'] -- the
        # repo's own, ~17 GB, is then never read. Tokenizer and processor stay base_edit's.
        te, te_src = _pick_text_encoder(base_edit, "edit")
        for name in ("scheduler", "vae", "text_encoder", "tokenizer", "processor"):
            if name == "text_encoder" and te is not None:
                comps[name] = te
                continue
            spec = mi.get(name)
            if not (isinstance(spec, list) and len(spec) == 2):
                continue
            lib, cls_name = spec
            Cls = getattr(importlib.import_module(lib), cls_name)
            kw = {"torch_dtype": DTYPE} if name in ("vae", "text_encoder") else {}
            # Load ONLY this subfolder (the encoder ~17 GB, the VAE...); the repo's
            # 40 GB transformer is never downloaded.
            comps[name] = Cls.from_pretrained(base_edit, subfolder=name, **kw)
        pipe = EditCls(**comps)
    else:
        # repo diffusers complet (telechargement). 2509 -> QwenImageEditPlusPipeline
        # (multi-images) ; revision de base -> QwenImageEditPipeline. Repli automatique.
        plus = "2509" in repo or "2511" in repo or "plus" in repo.lower()
        EditCls = (EditPlus if plus else None) or diffusers.QwenImageEditPipeline
        _log(f"loading Qwen-Image-Edit: {repo} via {EditCls.__name__} (offload={OFFLOAD_MODE}) ...")
        # A replacement encoder: compared against THIS repo's config, passed as
        # text_encoder= (diffusers then builds the pipe with it instead of loading the
        # repo's).
        te, te_src = _pick_text_encoder(repo, "edit")
        te_kw = {"text_encoder": te} if te is not None else {}
        try:
            pipe = EditCls.from_pretrained(repo, torch_dtype=DTYPE, **te_kw)
        except Exception as e:
            alt = diffusers.QwenImageEditPipeline
            if EditCls is alt:
                raise
            _log(f"{EditCls.__name__} failed ({e}); falling back to {alt.__name__}")
            pipe = alt.from_pretrained(repo, torch_dtype=DTYPE, **te_kw)
    _TEXT_ENCODER_ACTIVE_EDIT = te_src      # the pipe is mounted: says what really runs
    # The same rule as the base: a GGUF transformer only goes onto the GPU through
    # enable_model_cpu_offload (otherwise it stays on the CPU: ~800 s/step observed).
    _off = _effective_offload(repo)
    if _off != OFFLOAD_MODE:
        _log(f"GGUF edit: offload '{OFFLOAD_MODE}' forced to '{_off}' (a GGUF does not run "
             f"on the GPU in none/sequential -> otherwise CPU, ~800s/step)")
    if DEVICE == "cuda" and _off == "model":
        pipe.enable_model_cpu_offload()
    elif DEVICE == "cuda" and _off == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(DEVICE)
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception as e:
        _dbg(f"VAE tiling not available on edit pipe: {e}")
    _DERIVED["omni"] = pipe
    _log(f"Qwen-Image-Edit ready in {time.time() - t0:.1f}s")
    return pipe


@_gpu_serial
def generate_omni(refs, prompt, negative, width, height, steps, seed,
                  guidance=None, honor_size=False, steps_explicit=False):
    """Qwen-Image-Edit instruction-based editing: edits one (or several, through 2509)
    input image(s) according to the instruction prompt. Keeps upstream's signature (cz_ui).
    width/height are ignored by default (editing preserves the input's dimensions);
    honor_size=True passes them to the pipe (the edit protocol with an explicit size, e.g.
    the Upscaler preset which wants a 2x output). guidance = an override of the global CFG.
    The edit LoRAs (EDIT_LORAS, the 'Edit LoRAs' box) are applied hot here.
"""
    refs = [r.convert("RGB") for r in (refs or []) if r is not None]
    if not refs:
        raise ValueError("Edit needs at least one input image.")
    pipe = get_pipe("omni")
    _apply_edit_loras(pipe)
    # Fast mode: its steps/guidance win over Settings; a caller that set them
    # explicitly (protocol spec.steps / spec.guidance) keeps the upper hand.
    if EDIT_SPEED:
        if not steps_explicit:
            steps = EDIT_SPEED["steps"]
        if guidance is None:
            guidance = EDIT_SPEED["guidance"]
    g = float(GUIDANCE) if guidance is None else float(guidance)
    lora_info = ", edit LoRA " + "+".join(os.path.basename(p) for p, _ in _APPLIED_EDIT_LORAS) \
        if _APPLIED_EDIT_LORAS else ""            # the presets + the Lightning LoRA really applied
    _log(f"edit: {len(refs)} image(s), {int(steps)} steps, cfg {g:.1f}{lora_info} ...")
    _progress(0.1, f"Editing ({len(refs)} image(s))...")
    _set_slicing(pipe, max(max(r.size) for r in refs))
    t0 = time.time()
    # 2509/Plus takes a list of images; the base revision takes a single one.
    image_arg = refs if len(refs) > 1 else refs[0]
    size_kw = {}
    if honor_size and width and height:
        size_kw = {"width": round_to_multiple(int(width)), "height": round_to_multiple(int(height))}
        _dbg(f"edit: explicit output size {size_kw['width']}x{size_kw['height']}")
    out = _qwen_call(
        pipe,
        image=image_arg,
        prompt=prompt or "",
        num_inference_steps=int(steps),
        generator=_make_generator(seed),
        **size_kw,
        **_cfg(negative, guidance),
    ).images[0]
    _log(f"edit done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def load_pipe():
    """Compat: pipeline img2img (etage de raffinement)."""
    return get_pipe("img2img")


@_gpu_serial
def generate(prompt, width, height, steps, seed, negative_prompt=""):
    """Qwen-Image txt2img: generates an image from a prompt. Real CFG through
    true_cfg_scale (= the guidance slider, ~4.0), ~30-50 steps advised. The negative prompt
    works thanks to the real CFG (see _cfg)."""
    pipe = get_pipe("txt2img")
    w = round_to_multiple(int(width))
    h = round_to_multiple(int(height))
    _log(f"txt2img: {w}x{h}, {int(steps)} steps, cfg {GUIDANCE:.1f} ...")
    _dbg(f"txt2img seed={seed} dtype=bf16 device={DEVICE} offload={OFFLOAD_MODE} "
         f"transformer={'single-file' if ZIMAGE_TRANSFORMER else 'repo'}")
    if DEVICE == "cuda":
        _dbg(f"VRAM before: alloc={torch.cuda.memory_allocated()/1024**3:.2f} Go")
    _progress(0.1, f"Generating {w}x{h} ({int(steps)} steps)...")
    t0 = time.time()
    # Two attempts at most: when the VRAM guard fires at the first step ('none' mode
    # too optimistic), _consume_vram_downgrade switches to 'model' and we replay.
    for _attempt in (0, 1):
        _set_slicing(pipe, max(w, h))
        img = _qwen_call(
            pipe,
            prompt=prompt or "",
            width=w, height=h,
            num_inference_steps=int(steps),
            generator=_make_generator(seed),
            **_cfg(negative_prompt),
            **_vram_guard_kwargs(),
        ).images[0]
        if not _consume_vram_downgrade():
            break
        pipe = get_pipe("txt2img")   # reload with the downgraded offload
    _log(f"txt2img done in {time.time() - t0:.1f}s")
    if DEVICE == "cuda":
        _dbg(f"VRAM peak: alloc={torch.cuda.max_memory_allocated()/1024**3:.2f} Go | "
             f"reserved={torch.cuda.max_memory_reserved()/1024**3:.2f} Go")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return img


def round_to_multiple(x, m=16):
    return max(m, int(round(x / m) * m))


def set_force_ratio(spec):
    """Sets the forced ratio for upscale/img2img: 'W:H' / 'WxH' (e.g. '13:19',
    '832x1216') or '' to turn it off (the native ratio is preserved). Driven by the UI
    radio."""
    global FORCE_RATIO
    FORCE_RATIO = (spec or "").strip()
    _log(f"force ratio -> {FORCE_RATIO or '(off, ratio natif preserve)'}")


def set_force_ratio_mode(mode):
    """'crop' (a center crop) or 'extend' (outpainting the missing bands)."""
    global FORCE_RATIO_MODE
    FORCE_RATIO_MODE = "extend" if str(mode or "").strip().lower() == "extend" else "crop"
    _log(f"force ratio mode -> {FORCE_RATIO_MODE}")


def _parse_ratio(spec):
    """(w, h) from 'W:H', 'WxH', or a '832 x 1216 | 13:19' label; otherwise None."""
    import re
    if not spec:
        return None
    m = re.search(r"(\d+)\s*[:xX×]\s*(\d+)", str(spec))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (a, b) if a > 0 and b > 0 else None


def _crop_to_ratio(image, ratio_w, ratio_h):
    """Recadre (centre) l'image au ratio ratio_w:ratio_h en gardant l'aire maximale."""
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur > target:                       # too wide -> cut the sides
        nw = max(1, int(round(h * target)))
        x0 = (w - nw) // 2
        return image.crop((x0, 0, x0 + nw, h))
    nh = max(1, int(round(w / target)))    # trop haut -> couper haut/bas
    y0 = (h - nh) // 2
    return image.crop((0, y0, w, y0 + nh))


def _extend_to_ratio(image, ratio_w, ratio_h, prompt, steps, seed):
    """Brings the image to the target ratio by EXTENDING it (outpaint) instead of
    cropping: symmetric bands are added on the missing axis and filled by the model through
    outpaint_directions -- the centre keeps its full resolution (only the bands are
    generated, with the diffusion bounded to ~1 MP then recomposed).

    Anti 'banding': a light img2img pass (EXTEND_DENOISE) runs on the extended image, but
    ONLY the bands + a feathered transition margin are pasted back from that pass -- the
    original centre stays PIXEL FOR PIXEL intact (the pass harmonises exposure/texture at
    the seams without ever retouching the image).
"""
    from PIL import ImageDraw, ImageFilter
    image = image.convert("RGB")
    w, h = image.size
    target = float(ratio_w) / float(ratio_h)
    cur = w / h
    if abs(cur - target) < 1e-3:
        return image
    if cur < target:                       # too narrow -> widen left + right
        pad = target * h - w
        out = outpaint_directions(image, None, ["left", "right"], prompt, steps, seed,
                                  expand=pad / (2.0 * w))
    else:                                  # trop large -> etendre haut + bas
        pad = w / target - h
        out = outpaint_directions(image, None, ["top", "bottom"], prompt, steps, seed,
                                  expand=pad / (2.0 * h))
    if EXTEND_DENOISE > 0.001:
        _log(f"extend: seam-blend pass (img2img denoise {EXTEND_DENOISE:.2f}, "
             "original centre kept)")
        refined = _refine_whole(get_pipe("img2img"), out, EXTEND_DENOISE,
                                steps, prompt, seed)
        # Paste-back mask: white = take the harmonised pass (the bands + a transition
        # margin STRADDLING the seam), black = keep the original. The margin reaches into
        # the original image then is feathered -> a blended join, an intact centre.
        ox, oy = (out.width - w) // 2, (out.height - h) // 2
        m = max(24, int(0.05 * min(out.size)))       # a transition ~5% of the short side
        mx, my = (m if ox > 0 else 0), (m if oy > 0 else 0)   # a margin on the seam side ONLY
        mask = Image.new("L", out.size, 255)
        ImageDraw.Draw(mask).rectangle(
            [ox + mx, oy + my, ox + w - mx, oy + h - my], fill=0)
        mask = mask.filter(ImageFilter.GaussianBlur(max(8, m // 3)))
        out = Image.composite(refined, out, mask)
    return out


def _reframe_canvas(image, ratio_w, ratio_h, overlap=8):
    """Places the image in a larger canvas at the target ratio (expansion on 1 axis),
    + a mask (white = to fill, black = to keep, with a small overlap)."""
    from PIL import ImageDraw
    image = image.convert("RGB")
    w, h = image.size
    r = ratio_w / ratio_h
    # Aligned on 32 (patch 2 x VAE 16): avoids conv errors (no engine).
    if w / h < r:  # trop etroit -> elargir
        nw, nh = round_to_multiple(int(round(h * r)), 32), round_to_multiple(h, 32)
    else:          # too wide -> grow in height
        nw, nh = round_to_multiple(w, 32), round_to_multiple(int(round(w / r)), 32)
    nw, nh = max(nw, round_to_multiple(w, 32)), max(nh, round_to_multiple(h, 32))
    ox, oy = (nw - w) // 2, (nh - h) // 2
    canvas = Image.new("RGB", (nw, nh), (127, 127, 127))
    canvas.paste(image, (ox, oy))
    mask = Image.new("L", (nw, nh), 255)
    ImageDraw.Draw(mask).rectangle(
        [ox + overlap, oy + overlap, ox + w - overlap, oy + h - overlap], fill=0)
    return canvas, mask, nw, nh


@_gpu_serial
def inpaint_run(background, mask, prompt, steps, denoise, seed):
    """Inpaint: regenerates the white area of the mask according to the prompt
    (ZImageInpaintPipeline). background + mask = PIL (L: white = to change)."""
    orig = background.convert("RGB")
    full_mask = mask
    # Diffusion bounded to ~1 MP (the model's sweet spot), then recomposed at full
    # resolution.
    bg, work_mask, orig_size = _cap_work_res(orig, mask)
    w, h = bg.size
    pipe = get_pipe("inpaint")
    _log(f"inpaint: work {w}x{h} (orig {orig_size[0]}x{orig_size[1]}), {int(steps)} steps, "
         f"strength {float(denoise):.2f}, cfg {GUIDANCE:.1f} ...")
    _progress(0.1, "Inpainting...")
    _set_slicing(pipe, max(w, h))
    t0 = time.time()
    out = _qwen_call(pipe, prompt=prompt or "", image=bg, mask_image=work_mask,
                     strength=float(denoise), num_inference_steps=int(steps),
                     generator=_make_generator(seed), **_cfg(None)).images[0]
    # Recompose: outside the mask keeps the full resolution; the join is feathered.
    out = _composite_back(out, orig, full_mask, orig_size,
                          feather=max(2, int(min(orig_size) * 0.01)))
    _log(f"inpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# The Z-Image model's "sweet spot" target resolution (~1 MP, like the txt2img ratios).
# The reframe aims at that budget so as NOT to blow up the pixel count (a 2-3 MP output that
# leaves the training zone -> slow and degraded quality).
MODEL_TARGET_PX = 1024 * 1024


def _ratio_canvas(ratio_w, ratio_h, target_px=MODEL_TARGET_PX):
    """Dimensions (multiples de 32) d'un canevas au ratio donne, a ~target_px pixels."""
    r = float(ratio_w) / float(ratio_h)
    nh = (target_px / r) ** 0.5
    nw = nh * r
    return round_to_multiple(int(round(nw)), 32), round_to_multiple(int(round(nh)), 32)


def _cap_work_res(image, mask, max_px=MODEL_TARGET_PX):
    """Bounds the working resolution for the diffusion: when image > max_px, returns a
    reduced version (multiples of 32) of (image, mask) + the original size to recompose
    afterwards. Avoids running the model far above its sweet spot (~1 MP) -> faster and
    better quality."""
    w, h = image.size
    if w * h > max_px:
        s = (max_px / (w * h)) ** 0.5
        ww, wh = round_to_multiple(int(w * s), 32), round_to_multiple(int(h * s), 32)
    else:
        ww, wh = round_to_multiple(w, 32), round_to_multiple(h, 32)
    img_w = image.resize((ww, wh), Image.LANCZOS) if (ww, wh) != image.size else image
    msk_w = mask.resize((ww, wh), Image.NEAREST) if mask.size != (ww, wh) else mask
    return img_w, msk_w, (w, h)


def _composite_back(result, original, mask, orig_size, feather=0):
    """Recomposes at the original resolution: the masked area (white) comes from `result`
    (scaled back up to orig_size), the rest from `original` -> everything outside the mask
    keeps the starting image's full resolution. `feather` (px) blurs the mask to blend the
    join (a gradual original <-> generated transition, no hard line)."""
    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)
    if original.size != orig_size:
        original = original.resize(orig_size, Image.LANCZOS)
    m = (mask.resize(orig_size, Image.NEAREST) if mask.size != orig_size else mask).convert("L")
    if feather and feather > 0:
        from PIL import ImageFilter
        m = m.filter(ImageFilter.GaussianBlur(float(feather)))
    return Image.composite(result, original.convert("RGB"), m)


def reframe(image, ratio_w, ratio_h, fit, prompt, steps, seed, strength=1.0):
    """Crops the image to the target ratio while bounding the output to the model's sweet
    spot (~1 MP) -> no more pixel-count explosion.
      fit='contain' : the whole image fits inside the canvas (without enlarging it), and the
                      added edges are filled by Z-Image (outpaint).
      fit='cover'   : the image fills the canvas at the ratio then is center-cropped (no
                      outpaint, a plain reframe/crop).
"""
    from PIL import ImageDraw
    img = image.convert("RGB")
    w, h = img.size
    nw, nh = _ratio_canvas(ratio_w, ratio_h)
    if str(fit).lower() == "cover":
        scale = max(nw / w, nh / h)
        rw2, rh2 = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((rw2, rh2), Image.LANCZOS)
        left, top = (rw2 - nw) // 2, (rh2 - nh) // 2
        out = resized.crop((left, top, left + nw, top + nh))
        _log(f"reframe cover: {w}x{h} -> {nw}x{nh} (crop, no fill)")
        return out
    # contain -> the original is fitted without being enlarged, then the edges are
    # outpainted.
    from PIL import ImageFilter
    scale = min(nw / w, nh / h, 1.0)
    rw2, rh2 = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((rw2, rh2), Image.LANCZOS) if (rw2, rh2) != (w, h) else img
    ox, oy = (nw - rw2) // 2, (nh - rh2) // 2
    # Edges = a blurred extension of the edge colours (blurred edge fill, like the
    # outpaint) rather than a grey -> exposure continuity; it shows through when
    # strength < 1.0.
    arr = np.pad(np.array(resized), [[oy, nh - rh2 - oy], [ox, nw - rw2 - ox], [0, 0]],
                 mode="edge")
    canvas = Image.fromarray(np.ascontiguousarray(arr))
    overlap = 8
    mask = Image.new("L", (nw, nh), 255)
    ImageDraw.Draw(mask).rectangle(
        [ox + overlap, oy + overlap, ox + rw2 - overlap, oy + rh2 - overlap], fill=0)
    blur_r = max(8, int(min(nw, nh) * 0.03))
    canvas = Image.composite(canvas.filter(ImageFilter.GaussianBlur(blur_r)), canvas, mask)
    pipe = get_pipe("inpaint")
    _log(f"reframe contain (outpaint): {w}x{h} -> {nw}x{nh}, {int(steps)} steps, "
         f"strength {float(strength):.2f}, cfg {GUIDANCE:.1f} ...")
    _progress(0.1, f"Reframe -> {nw}x{nh}...")
    _set_slicing(pipe, max(nw, nh))
    t0 = time.time()
    out = _qwen_call(pipe, prompt=prompt or "", image=canvas, mask_image=mask,
                     strength=float(strength), num_inference_steps=int(steps),
                     generator=_make_generator(seed), **_cfg(None)).images[0]
    if out.size != (nw, nh):
        out = out.resize((nw, nh), Image.LANCZOS)
    _log(f"reframe done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


@_gpu_serial
def outpaint(image, ratio_w, ratio_h, prompt, steps, seed):
    """Compat (CLI --reframe and existing calls): a reframe in 'contain' mode (outpaint),
    bounded to the model's sweet spot."""
    return reframe(image, ratio_w, ratio_h, "contain", prompt, steps, seed)


def outpaint_directions(image, mask, directions, prompt, steps, seed, strength=1.0, expand=0.3):
    """Directional outpaint (Fooocus-style): enlarges the image in the chosen directions
    among left/right/top/bottom, each by `expand` (a fraction of the original dimension), by
    replicating the edge pixels (mode 'edge'), then has the added bands filled by Z-Image
    (ZImageInpaintPipeline). A painted `mask` (L, white = to change) is optional: it is kept
    over the original area and combined with the added bands (white)."""
    img = np.array(image.convert("RGB"))
    H, W = img.shape[:2]
    m = np.array(mask.convert("L")) if mask is not None else np.zeros((H, W), dtype=np.uint8)
    dirs = set(d.lower() for d in (directions or []))
    if "top" in dirs:
        p = int(H * expand)
        img = np.pad(img, [[p, 0], [0, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[p, 0], [0, 0]], mode="constant", constant_values=255)
    if "bottom" in dirs:
        p = int(H * expand)
        img = np.pad(img, [[0, p], [0, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[0, p], [0, 0]], mode="constant", constant_values=255)
    if "left" in dirs:
        p = int(W * expand)
        img = np.pad(img, [[0, 0], [p, 0], [0, 0]], mode="edge")
        m = np.pad(m, [[0, 0], [p, 0]], mode="constant", constant_values=255)
    if "right" in dirs:
        p = int(W * expand)
        img = np.pad(img, [[0, 0], [0, p], [0, 0]], mode="edge")
        m = np.pad(m, [[0, 0], [0, p]], mode="constant", constant_values=255)
    canvas = Image.fromarray(np.ascontiguousarray(img))
    mask_img = Image.fromarray(np.ascontiguousarray(m))
    full_size = canvas.size
    # Dilate the area to generate a little towards the inside -> the model regenerates
    # a thin transition band that joins up with the original (avoids a hard seam).
    from PIL import ImageFilter
    k = max(3, (int(min(full_size) * 0.02) // 2) * 2 + 1)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(min(k, 15)))
    # "Blurred edge fill": the area to generate is filled with a BLURRED version of
    # the edge extension (the same colours/tone as the original) instead of a sharp
    # replicated edge. With strength < 1.0 that blur shows through -> exposure continuity
    # (no lighter band any more) and the model adds the detail on top.
    blur_r = max(8, int(min(full_size) * 0.03))
    canvas = Image.composite(canvas.filter(ImageFilter.GaussianBlur(blur_r)), canvas, mask_img)
    # Diffusion bounded to ~1 MP (the sweet spot), then recomposed: the centre (the
    # original image) keeps its full resolution, only the added edges are generated.
    work_img, work_mask, _ = _cap_work_res(canvas, mask_img)
    w2, h2 = work_img.size
    pipe = get_pipe("inpaint")
    _log(f"outpaint {sorted(dirs)}: {image.size[0]}x{image.size[1]} -> "
         f"{full_size[0]}x{full_size[1]} (work {w2}x{h2}), {int(steps)} steps, "
         f"cfg {GUIDANCE:.1f} ...")
    _progress(0.1, f"Outpaint -> {full_size[0]}x{full_size[1]}...")
    _set_slicing(pipe, max(w2, h2))
    t0 = time.time()
    out = _qwen_call(pipe, prompt=prompt or "", image=work_img, mask_image=work_mask,
                     strength=float(strength), num_inference_steps=int(steps),
                     generator=_make_generator(seed), **_cfg(None)).images[0]
    out = _composite_back(out, canvas, mask_img, full_size,
                          feather=max(4, int(min(full_size) * 0.015)))
    _log(f"outpaint done in {time.time() - t0:.1f}s")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def _make_generator(seed):
    return torch.Generator(DEVICE).manual_seed(int(seed)) if int(seed) >= 0 else None


@_gpu_serial
def _refine_whole(pipe, image, denoise, steps, prompt, seed):
    """A Qwen-Image img2img pass over the whole image (or over one tile). The slicing is
    set according to the size really processed: a 1024 tile -> OFF (fast), whole 2K+ -> ON.
    IMPORTANT: width/height = the image's size (aligned on 16) are passed. Otherwise
    Qwen-Image img2img falls back to its default (height = default_sample_size *
    vae_scale_factor = 1024) and RESIZES the input to 1024x1024 -> the ratio is crushed (a
    bug). Forcing the input's dimensions preserves the original ratio in upscale/img2img."""
    w = round_to_multiple(image.width, 16)
    h = round_to_multiple(image.height, 16)
    # Two attempts at most: the VRAM guard at the first step (see generate), then a retry in 'model'.
    for _attempt in (0, 1):
        _set_slicing(pipe, max(image.size))   # a reposer sur le pipe recharge du retry
        out = _qwen_call(
            pipe,
            prompt=prompt or "",
            image=image,
            width=w, height=h,
            strength=float(denoise),
            num_inference_steps=int(steps),
            generator=_make_generator(seed),
            **_cfg(None),
            **_vram_guard_kwargs(),
        ).images[0]
        if not _consume_vram_downgrade():
            return out
        pipe = get_pipe("img2img")   # reload with the downgraded offload
    return out


def _feather_mask_np(th, tw, overlap, left, right, top, bottom):
    """A (th, tw, 1) mask with a linear ramp on the edges that adjoin another tile."""
    mask = np.ones((th, tw, 1), dtype=np.float32)
    f = int(overlap)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        if left:
            mask[:, :f, 0] *= ramp[np.newaxis, :]
        if right:
            mask[:, tw - f:, 0] *= ramp[::-1][np.newaxis, :]
        if top:
            mask[:f, :, 0] *= ramp[:, np.newaxis]
        if bottom:
            mask[th - f:, :, 0] *= ramp[::-1][:, np.newaxis]
    return mask


def _refine_tiled(pipe, image, denoise, steps, prompt, seed, tile, overlap):
    """A Z-Image pass in tiles with feathered recomposition (Ultimate SD Upscale style).
    Caps the VRAM peak (one tile at a time) and makes 4K+ possible without seams.
    The same linear ramp + overlap-add as esrgan_upscale, but at scale 1 on PIL."""
    w, h = image.size
    tile = round_to_multiple(tile)                       # a multiple of 16 for the VAE
    overlap = max(0, min(int(overlap), tile - 16))
    if w <= tile and h <= tile:
        # A single tile = the whole image -> no duplication possible: the requested
        # denoise.
        return _refine_whole(pipe, image, denoise, steps, prompt, seed)
    # Anti-duplication 1: an empty prompt per tile (the global prompt describes the
    # whole composition).
    prompt = _tile_prompt(prompt)
    if not (prompt or "").strip():
        _log("refine tiled: empty prompt per tile (anti-duplication; rule refine_tile_prompt).")
    # Anti-duplication 2 (a safety net): at a high denoise each tile can still drift.
    denoise = float(denoise)
    if _TILE_DENOISE_CAP > 0 and denoise > _TILE_DENOISE_CAP:
        _log(f"refine tiled: denoise {denoise:.2f} > plafond {_TILE_DENOISE_CAP:.2f} -> "
             f"reduit a {_TILE_DENOISE_CAP:.2f} (regle refine_tile_denoise_cap).")
        denoise = _TILE_DENOISE_CAP

    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    step = max(16, tile - overlap)
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    total = len(ys) * len(xs)
    _log(f"refine: tiled {w}x{h}, tile {tile} overlap {overlap} -> {len(xs)}x{len(ys)} = {total} tiles")
    i = 0
    for y in ys:
        for x in xs:
            if _STOP:
                _log("refine tiled: stop requested")
                break
            i += 1
            x2, y2 = min(x + tile, w), min(y + tile, h)
            x1, y1 = max(x2 - tile, 0), max(y2 - tile, 0)
            cw, ch = x2 - x1, y2 - y1
            _progress(0.45 + 0.5 * (i - 1) / max(1, total), f"Refine tile {i}/{total}")
            crop = image.crop((x1, y1, x2, y2))
            _t_tile = time.time()
            out = _refine_whole(pipe, crop, denoise, steps, prompt, seed)
            _log(f"  tile {i}/{total} ({cw}x{ch}) in {time.time() - _t_tile:.1f}s{_vram_str()}")
            if out.size != (cw, ch):
                out = out.resize((cw, ch), Image.LANCZOS)
            out_arr = np.asarray(out.convert("RGB"), dtype=np.float32) / 255.0
            mask = _feather_mask_np(ch, cw, overlap,
                                    left=x1 > 0, right=x2 < w, top=y1 > 0, bottom=y2 < h)
            acc[y1:y2, x1:x2, :] += out_arr * mask
            weight[y1:y2, x1:x2, :] += mask

    out = acc / np.clip(weight, 1e-6, None)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


# ----------------------------------------------------------------------------
# Orchestration: process_one, the txt2img batch (run/_gen_meta stay in app.py because
# run emits gr.Error for the UI).
# ----------------------------------------------------------------------------
@_gpu_serial
def process_one(image, esrgan_model, factor, denoise, steps, prompt, seed, tile, overlap,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                do_esrgan=True, refine_first=False, apply_force_ratio=False):
    """Pipeline over one PIL Image, returns (image, timings_dict).
    do_esrgan=False -> pure img2img (skips the ESRGAN stage, refines the native image).
    refine_first=True -> refine THEN ESRGAN (the diffusion runs at the native resolution =
    far faster), instead of ESRGAN THEN refine (detail at high resolution).
    apply_force_ratio=True + FORCE_RATIO set -> brings the INPUT to the chosen ratio before
    processing: FORCE_RATIO_MODE 'crop' = a center crop (Fooocus-style), 'extend' =
    outpaints the missing bands (nothing is lost). Otherwise: the native ratio is preserved.
"""
    timings = {"esrgan": 0.0, "refine": 0.0}
    image = image.convert("RGB")
    if apply_force_ratio and FORCE_RATIO:
        r = _parse_ratio(FORCE_RATIO)
        if r:
            _before = image.size
            if FORCE_RATIO_MODE == "extend":
                # max(6, steps): outpainting the bands stays correct even when the
                # upscale runs as pure ESRGAN (steps/denoise at ~0).
                image = _extend_to_ratio(image, r[0], r[1], prompt, max(6, int(steps)), seed)
                _verb = "extend (outpaint)"
            else:
                image = _crop_to_ratio(image, r[0], r[1])
                _verb = "crop"
            _log(f"force ratio {r[0]}:{r[1]} -> {_verb} {_before[0]}x{_before[1]} "
                 f"to {image.size[0]}x{image.size[1]}")
    w0, h0 = image.size
    use_esrgan = bool(do_esrgan and esrgan_model)
    do_refine = float(denoise) > 0.001
    _dbg(f"process_one in={w0}x{h0} factor={factor} denoise={denoise} steps={int(steps)} "
         f"do_esrgan={do_esrgan} refine_first={refine_first} esrgan={esrgan_model} "
         f"refine_tile={int(refine_tile)}")

    def _esrgan_stage(img):
        t0 = time.time()
        iw, ih = img.size
        _progress(0.15, f"ESRGAN upscale {iw}x{ih}...")
        model = load_esrgan(esrgan_model)
        _log(f"ESRGAN upscale: {iw}x{ih} (tile {int(tile)}) ...")
        up = esrgan_upscale(img, model, int(tile), int(overlap))
        # The target = the factor applied to the original size (order-independent).
        target_w = round_to_multiple(w0 * factor)
        target_h = round_to_multiple(h0 * factor)
        up = up.resize((target_w, target_h), Image.LANCZOS)
        timings["esrgan"] += time.time() - t0
        _log(f"ESRGAN done in {timings['esrgan']:.1f}s -> {target_w}x{target_h}")
        return up

    def _refine_stage(img):
        t0 = time.time()
        pipe = load_pipe()
        rw, rh = img.size
        rt = int(refine_tile)
        # Anti-crash guard rail: a whole-image refine that is too large (4K+) -> auto-tiling.
        if rt <= 0 and max(rw, rh) > _AUTO_TILE_ABOVE:
            rt = _pick_refine_tile(rw, rh, int(refine_overlap) or 64)
            _log(f"refine: image {rw}x{rh} > {_AUTO_TILE_ABOVE}px -> auto-tiling (tile {rt}) "
                 "to avoid the VRAM spike (rules: auto_refine_tile_above, auto_refine_tile)")
        if rt > 0:
            out = _refine_tiled(pipe, img, denoise, steps, prompt, seed,
                                rt, int(refine_overlap) or 64)
        else:
            _log(f"Qwen refine: whole image {rw}x{rh}, denoise {float(denoise):.2f}, "
                 f"{int(steps)} steps ...")
            _progress(0.5, f"Qwen refine {rw}x{rh}...")
            out = _refine_whole(pipe, img, denoise, steps, prompt, seed)
        timings["refine"] += time.time() - t0
        return out

    result = image
    if refine_first:
        # refine on the native image (fast) then the ESRGAN enlargement.
        if do_refine:
            result = _refine_stage(result)
        if use_esrgan:
            result = _esrgan_stage(result)
    else:
        # the classic order: ESRGAN (the detailer) then refine at the enlarged resolution.
        if use_esrgan:
            result = _esrgan_stage(result)
        if do_refine:
            result = _refine_stage(result)

    if not use_esrgan and not do_refine:
        _log(f"process_one: nothing to do (no ESRGAN, denoise=0) on {w0}x{h0}")

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    _progress(1.0, "Done")
    _log(f"process_one done | esrgan {timings['esrgan']:.1f}s + refine {timings['refine']:.1f}s "
         f"= {timings['esrgan'] + timings['refine']:.1f}s")
    return result, timings


@_gpu_serial
def txt2img_run(prompt, width, height, gen_steps, seed, negative_prompt="",
                upscale=False, esrgan_model=None, factor=2.0, denoise=0.30, steps=12,
                tile=DEFAULT_TILE, overlap=DEFAULT_OVERLAP,
                refine_tile=DEFAULT_REFINE_TILE, refine_overlap=DEFAULT_REFINE_OVERLAP,
                refine_first=False):
    """Generates an image (Z-Image txt2img) then, when upscale=True, runs it through the
    ESRGAN + refine pipeline. Returns (image, timings_dict)."""
    timings = {"txt2img": 0.0, "esrgan": 0.0, "refine": 0.0}
    t0 = time.time()
    base = generate(prompt, width, height, gen_steps, seed, negative_prompt)
    timings["txt2img"] = time.time() - t0
    if not upscale:
        return base, timings
    result, t = process_one(base, esrgan_model, factor, denoise, steps, prompt, seed,
                            tile, overlap, refine_tile=refine_tile, refine_overlap=refine_overlap,
                            refine_first=refine_first)
    timings["esrgan"] = t.get("esrgan", 0.0)
    timings["refine"] = t.get("refine", 0.0)
    return result, timings


def _gen_meta(mode, prompt, negative="", seed=None, steps=None, guidance=None,
              size=None, model=None, styles=None, extra=None):
    """Builds the generation metadata dict (for the sidecar/PNG)."""
    m = {"app": "crispz-qwen-edit", "mode": mode, "prompt": prompt or "",
         "negative": negative or "", "date": _now_stamp()}
    if seed is not None and int(seed) >= 0:
        m["seed"] = int(seed)
    if steps is not None:
        m["steps"] = int(steps)
    if guidance is not None:
        m["guidance"] = float(guidance)
    if size:
        m["size"] = f"{size[0]}x{size[1]}"
    # Names of the applied styles (on top of the keywords already injected into the
    # prompt).
    _styles = [s for s in (styles or []) if s and s not in ("None", "none")]
    if _styles:
        m["styles"] = _styles
    m["sampler"] = f"{SAMPLER}/{SCHEDULE}"
    m["model"] = model or (ZIMAGE_TRANSFORMER or BASE_REPO)
    # A replacement encoder: the one that REALLY ran, by its folder name.
    # An edit ('omni') comes out of the edit pipe, which is loaded separately: it has its own
    # state. Requested but dropped at load time = the image comes from the repo's encoder,
    # and the one that did not serve is named separately.
    _te = _TEXT_ENCODER_ACTIVE_EDIT if mode == "omni" else _TEXT_ENCODER_ACTIVE
    if _te:
        m["text_encoder"] = _encoder_label(_te)
    elif TEXT_ENCODER:
        m["text_encoder_not_applied"] = _encoder_label(TEXT_ENCODER)
    if LORAS:
        m["loras"] = [f"{os.path.basename(p)}@{w}" for p, w in LORAS]
    if extra:
        m.update(extra)
    return m
