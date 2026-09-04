"""Registre des LoRA d'EDITION Qwen-Image-Edit (presets "fast lazy load").

Source: github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load
(ADAPTER_SPECS). Chaque preset = un LoRA Hugging Face entraine pour une tache
d'edition (photo -> anime, relighting, upscale 2K, angles de camera...). Ils
n'ont PAS de trigger word: l'instruction en langage naturel suffit
(`prompt` ci-dessous = exemple de l'upstream).

Chargement paresseux: rien n'est telecharge a l'import. `resolve(name)` rend
le chemin local du .safetensors et le telecharge depuis le hub a la premiere
demande, dans `<LORAS_DIR>/_hf-edit/<adapter_name>.safetensors` (nom ASCII
stable: plusieurs fichiers upstream ont des noms chinois ou des espaces).
Une fois sur disque, c'est un LoRA ordinaire: `--lora`, `<lora:...>`,
`caps.loras` et le dropdown LoRA du base le voient sans code special.

Ces LoRA visent le pipe d'EDITION (cz_pipeline.generate_omni), pas le
txt2img: cz_pipeline garde un jeu separe (EDIT_LORAS / set_edit_loras).

Surcharge possible dans config.txt:
    "edit_loras_dir": "",          # dossier des telechargements (defaut <loras_dir>/_hf-edit)
    "edit_loras": {"Mon-Preset": {"repo": "...", "weights": "x.safetensors",
                                  "adapter_name": "mon-preset", "prompt": "...",
                                  "inputs": 1}, "Anime-V2": null}   # null = retire
"""
import os

from cz_core import CONFIG, _log, _dbg

# Ordre = ordre du dropdown. inputs = nombre d'images attendu (2 = input + reference).
EDIT_LORA_SPECS = {
    "Multiple-Angles": {
        "repo": "dx8152/Qwen-Edit-2509-Multiple-angles",
        "weights": "镜头转换.safetensors",
        "adapter_name": "multiple-angles",
        "prompt": "Rotate the camera 45 degrees to the right.",
        "inputs": 1, "base": "2509"},
    "Photo-to-Anime": {
        "repo": "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
        "weights": "Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
        "adapter_name": "photo-to-anime",
        "prompt": "Transform into anime.",
        "inputs": 1, "base": "2509"},
    "Anime-V2": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Anime",
        "weights": "Qwen-Image-Edit-2511-Anime-2000.safetensors",
        "adapter_name": "anime-v2",
        "prompt": "Transform into anime (while preserving the background and remaining "
                  "elements maintaining realism and original details.)",
        "inputs": 1, "base": "2511"},
    "Light-Migration": {
        "repo": "dx8152/Qwen-Edit-2509-Light-Migration",
        "weights": "参考色调.safetensors",
        "adapter_name": "light-migration",
        "prompt": "Refer to the color tone, remove the original lighting from Image 1, "
                  "and relight Image 1 based on the lighting and color tone of Image 2.",
        "inputs": 2, "base": "2509"},
    "Upscaler": {
        "repo": "starsfriday/Qwen-Image-Edit-2511-Upscale2K",
        "weights": "qwen_image_edit_2511_upscale.safetensors",
        "adapter_name": "upscale-2k",
        "prompt": "Upscale this picture to 4K resolution.",
        "inputs": 1, "base": "2511"},
    "Style-Transfer": {
        "repo": "zooeyy/Style-Transfer",
        "weights": "Style Transfer-Alpha-V0.1.safetensors",
        "adapter_name": "style-transfer",
        "prompt": "Convert Image 1 to the style of Image 2.",
        "inputs": 2, "base": "2511"},
    "Manga-Tone": {
        "repo": "nappa114514/Qwen-Image-Edit-2509-Manga-Tone",
        "weights": "tone001.safetensors",
        "adapter_name": "manga-tone",
        "prompt": "Paint with manga tone.",
        "inputs": 1, "base": "2509"},
    "Anything2Real": {
        "repo": "lrzjason/Anything2Real_2601",
        "weights": "anything2real_2601.safetensors",
        "adapter_name": "anything2real",
        "prompt": "Change the picture to realistic photograph.",
        "inputs": 1, "base": "2511"},
    "Fal-Multiple-Angles": {
        "repo": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        "weights": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
        "adapter_name": "fal-multiple-angles",
        "prompt": "Front-right quarter view.",
        "inputs": 1, "base": "2511"},
    "Polaroid-Photo": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo",
        "weights": "Qwen-Image-Edit-2511-Polaroid-Photo.safetensors",
        "adapter_name": "polaroid-photo",
        "prompt": "cinematic polaroid with soft grain subtle vignette gentle lighting white "
                  "frame handwritten photographed preserving realistic texture and details.",
        "inputs": 1, "base": "2511"},
    "Unblur-Anything": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale",
        "weights": "Qwen-Image-Edit-Unblur-Upscale_15.safetensors",
        "adapter_name": "unblur-anything",
        "prompt": "Unblur and upscale.",
        "inputs": 1, "base": "2511"},
    "Midnight-Noir-Eyes-Spotlight": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight",
        "weights": "Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight.safetensors",
        "adapter_name": "midnight-noir-eyes-spotlight",
        "prompt": "Transform into Midnight Noir Eyes Spotlight.",
        "inputs": 1, "base": "2511"},
    "Hyper-Realistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        "weights": "HRP_20.safetensors",
        "adapter_name": "hyper-realistic-portrait",
        "prompt": "Transform into a hyper-realistic face portrait.",
        "inputs": 1, "base": "2511"},
    "Ultra-Realistic-Portrait": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
        "weights": "URP_20.safetensors",
        "adapter_name": "ultra-realistic-portrait",
        "prompt": "Ultra-realistic portrait.",
        "inputs": 1, "base": "2511"},
    "Pixar-Inspired-3D": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Pixar-Inspired-3D",
        "weights": "PI3_20.safetensors",
        "adapter_name": "pixar-inspired-3d",
        "prompt": "Transform it into Pixar-inspired 3D.",
        "inputs": 1, "base": "2511"},
    "Noir-Comic-Book": {
        "repo": "prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel",
        "weights": "Noir-Comic-Book-Panel_20.safetensors",
        "adapter_name": "noir-comic-book",
        "prompt": "Transform into a noir comic book style.",
        "inputs": 1, "base": "2511"},
    "Any-Light": {
        "repo": "lilylilith/QIE-2511-MP-AnyLight",
        "weights": "QIE-2511-AnyLight_.safetensors",
        "adapter_name": "any-light",
        "prompt": "Apply the lighting from image 2 to image 1.",
        "inputs": 2, "base": "2511"},
    "Studio-DeLight": {
        "repo": "prithivMLmods/QIE-2511-Studio-DeLight",
        "weights": "QIE-2511-Studio-DeLight-5000.safetensors",
        "adapter_name": "studio-delight",
        "prompt": "Neutral uniform lighting. Preserve identity and composition.",
        "inputs": 1, "base": "2511"},
    "Cinematic-FlatLog": {
        "repo": "prithivMLmods/QIE-2511-Cinematic-FlatLog-Control",
        "weights": "QIE-2511-Cinematic-FlatLog-Control-3200.safetensors",
        "adapter_name": "flat-log",
        "prompt": "Transform into a cinematic flat log.",
        "inputs": 1, "base": "2511"},
}

SUBDIR = "_hf-edit"


def _apply_config_overrides(specs):
    """config 'edit_loras': dict nom -> spec (ajout/remplacement) ou null (retrait)."""
    extra = CONFIG.get("edit_loras")
    if not isinstance(extra, dict):
        return specs
    out = dict(specs)
    for name, spec in extra.items():
        if spec is None:
            out.pop(name, None)
        elif isinstance(spec, dict) and spec.get("repo") and spec.get("weights"):
            s = dict(spec)
            s.setdefault("adapter_name", name.lower().replace(" ", "-"))
            s.setdefault("prompt", "")
            s.setdefault("inputs", 1)
            out[name] = s
        else:
            _log(f"edit_loras['{name}'] ignored: needs 'repo' and 'weights'")
    return out


SPECS = _apply_config_overrides(EDIT_LORA_SPECS)


def names():
    """Noms des presets, dans l'ordre du registre."""
    return list(SPECS)


def spec(name):
    """Spec d'un preset (None si inconnu). Tolere l'adapter_name et la casse."""
    if not name:
        return None
    if name in SPECS:
        return SPECS[name]
    low = str(name).strip().lower()
    for n, s in SPECS.items():
        if n.lower() == low or s.get("adapter_name", "").lower() == low:
            return s
    return None


def canonical_name(name):
    """Nom du registre pour un nom/adapter_name (None si inconnu)."""
    s = spec(name)
    if s is None:
        return None
    for n, v in SPECS.items():
        if v is s:
            return n
    return None


def edit_loras_dir():
    """Dossier des LoRA d'edition telecharges: config 'edit_loras_dir', sinon
    <LORAS_DIR>/_hf-edit (LORAS_DIR lu a l'appel: l'UI peut le changer)."""
    d = (CONFIG.get("edit_loras_dir") or "").strip()
    if d:
        return d
    # LORAS_DIR SANS importer cz_pipeline (torch): caps du protocole doit rester
    # leger. Si le pipeline est deja charge (UI), sa valeur courante prime.
    import sys
    cp = sys.modules.get("cz_pipeline")
    base = getattr(cp, "LORAS_DIR", None) if cp is not None else None
    if not base:
        from cz_core import HERE, _prefs
        base = (os.environ.get("LORAS_DIR") or _prefs.get("loras_dir")
                or CONFIG.get("loras_dir") or os.path.join(HERE, "loras"))
    return os.path.join(base, SUBDIR)


def local_path(name):
    """Chemin local attendu du preset (existant ou non)."""
    s = spec(name)
    if s is None:
        return None
    return os.path.join(edit_loras_dir(), s["adapter_name"] + ".safetensors")


def is_downloaded(name):
    p = local_path(name)
    return bool(p) and os.path.isfile(p)


def resolve(name, download=True, progress=None):
    """Chemin local du LoRA du preset; le telecharge du hub si absent (et
    download=True). Leve FileNotFoundError si absent et download=False,
    RuntimeError si le telechargement echoue."""
    s = spec(name)
    if s is None:
        raise KeyError(f"unknown edit LoRA preset: {name!r} (known: {', '.join(SPECS)})")
    dst = local_path(name)
    if os.path.isfile(dst):
        return dst
    if not download:
        raise FileNotFoundError(dst)
    return _download(s, dst, progress)


def _download(s, dst, progress=None):
    """hf_hub_download -> copie atomique vers dst (nom ASCII stable)."""
    import shutil
    from huggingface_hub import hf_hub_download
    from cz_core import _apply_hf_token
    _apply_hf_token(CONFIG.get("hf_token"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    _log(f"edit LoRA: downloading {s['repo']}/{s['weights']} -> {dst}")
    if progress:
        progress(0.0, f"Downloading {s['repo']} ...")
    try:
        src = hf_hub_download(s["repo"], s["weights"])
    except Exception as e:
        raise RuntimeError(f"download failed for {s['repo']}/{s['weights']}: {e}") from e
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    _dbg(f"edit LoRA stored: {dst} ({os.path.getsize(dst) / 1024**2:.0f} MB)")
    if progress:
        progress(1.0, "Downloaded.")
    return dst


def status_label(name):
    """Libelle pour l'UI: 'Nom ✓' si deja sur disque, 'Nom ⬇' sinon."""
    return f"{name} {'✓' if is_downloaded(name) else '⬇'}"


def strip_label(label):
    """Inverse de status_label (le dropdown renvoie le libelle)."""
    if not label:
        return ""
    return str(label).rstrip(" ✓⬇").strip()


def catalog():
    """Liste legere pour caps / UI: [{name, adapter_name, repo, prompt, inputs,
    base, downloaded}] - sans rien telecharger."""
    return [{"name": n, "adapter_name": s["adapter_name"], "repo": s["repo"],
             "prompt": s.get("prompt", ""), "inputs": int(s.get("inputs", 1)),
             "base": s.get("base", ""), "downloaded": is_downloaded(n)}
            for n, s in SPECS.items()]
