"""Une edition doit etre encodee AVEC son image.

QwenImageEditPlusPipeline.encode_prompt(prompt, image) prefixe le prompt des jetons de
vision de l'image ('Picture 1: <|vision_start|><|image_pad|><|vision_end|>') et
l'encodeur Qwen2.5-VL lit l'image avec l'instruction. __call__ saute son propre
encodage des qu'on lui passe prompt_embeds. Or le cache d'embeddings (736e0ce)
appelait encode_prompt SANS image: chaque edition tournait sur un embedding texte
seul -- le transformer recevait bien les latents de l'image, l'encodeur jamais.

Regle: le cache est court-circuite quand l'appel porte une image ET que encode_prompt
du pipeline a un parametre `image`. img2img / inpaint recoivent aussi `image` (image de
depart) mais leur encode_prompt n'en a pas: ils gardent le cache, et le detailer son
gain.

Pipelines factices; les signatures des vraies classes diffusers sont lues sans charger
le moindre poids.

Run:  .venv/Scripts/python tests/test_edit_prompt_image.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_pipeline as P


class _Pipe:
    """Compte les encodages et retient ce que __call__ a recu."""

    _execution_device = "cpu"

    def __init__(self):
        self.text_encoder = object()
        self.encodes = []
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return type("o", (), {"images": ["IMG"]})()


class EditPipe(_Pipe):
    """encode_prompt prend l'image, comme QwenImageEdit(Plus)Pipeline."""

    def encode_prompt(self, prompt, image=None, device=None):
        self.encodes.append(image)
        return torch.zeros(1, 4, 8), torch.ones(1, 4)


class Img2ImgPipe(_Pipe):
    """encode_prompt sans image, comme QwenImageImg2Img / QwenImageInpaint."""

    def encode_prompt(self, prompt, device=None):
        self.encodes.append(None)
        return torch.zeros(1, 4, 8), torch.ones(1, 4)


def _fresh():
    P._embed_cache_clear()
    P._EMBED_CACHE_MAX = 8                      # config.txt peut l'avoir coupe


def test_an_edit_is_encoded_with_its_image():
    _fresh()
    pipe = EditPipe()
    for _ in range(2):
        P._qwen_call(pipe, image="REF", prompt="make the car red")
    assert pipe.encodes == [], "le cache a encode l'instruction sans l'image"
    for kw in pipe.calls:
        assert kw["prompt"] == "make the car red" and kw["image"] == "REF", kw
        assert not any(k in kw for k in P._EMBED_OUTS), "embeddings texte seul passes"
    assert not P._EMBED_CACHE, P._EMBED_CACHE
    print("OK test_an_edit_is_encoded_with_its_image")


def test_a_multi_reference_edit_too():
    _fresh()
    pipe = EditPipe()
    P._qwen_call(pipe, image=["A", "B"], prompt="put the hat from picture 2 on picture 1")
    assert pipe.encodes == [] and pipe.calls[0]["prompt"].startswith("put the hat")
    assert "prompt_embeds" not in pipe.calls[0]
    print("OK test_a_multi_reference_edit_too")


def test_img2img_and_inpaint_keep_the_cache():
    """Leur `image` est l'image de depart, pas une entree de l'encodeur."""
    _fresh()
    pipe = Img2ImgPipe()
    for _ in range(3):
        P._qwen_call(pipe, image="INIT", prompt="a lighthouse", strength=0.3)
    assert len(pipe.encodes) == 1, pipe.encodes
    for kw in pipe.calls:
        assert kw["prompt"] is None and kw["prompt_embeds"] is not None, kw
        assert kw["image"] == "INIT"
    print("OK test_img2img_and_inpaint_keep_the_cache")


def test_without_an_image_the_edit_pipe_may_use_the_cache():
    """Sans image, l'encodage EST texte seul: le reutiliser est juste."""
    _fresh()
    pipe = EditPipe()
    P._qwen_call(pipe, prompt="p")
    P._qwen_call(pipe, prompt="p")
    assert pipe.encodes == [None], pipe.encodes
    print("OK test_without_an_image_the_edit_pipe_may_use_the_cache")


def test_the_installed_diffusers_signatures():
    """La regle repose sur la signature: on la verifie sur les vraies classes."""
    from diffusers import (QwenImageEditPipeline, QwenImageEditPlusPipeline,
                           QwenImageImg2ImgPipeline, QwenImageInpaintPipeline,
                           QwenImagePipeline)
    assert P._encodes_with_image(QwenImageEditPlusPipeline)
    assert P._encodes_with_image(QwenImageEditPipeline)
    for cls in (QwenImagePipeline, QwenImageImg2ImgPipeline, QwenImageInpaintPipeline):
        assert not P._encodes_with_image(cls), cls.__name__
    print("OK test_the_installed_diffusers_signatures")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    _fresh()
    print("All edit-prompt-image tests passed.")
