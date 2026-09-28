"""An edit must be encoded WITH its image.

QwenImageEditPlusPipeline.encode_prompt(prompt, image) prefixes the prompt with the image's
vision tokens ('Picture 1: <|vision_start|><|image_pad|><|vision_end|>') and
the Qwen2.5-VL encoder reads the image along with the instruction. __call__ skips its own
encoding as soon as it is passed prompt_embeds. And yet the embeddings cache (736e0ce)
called encode_prompt WITHOUT the image: every edit ran on a text-only
embedding -- the transformer did receive the image's latents, the encoder never did.

The rule: the cache is short-circuited when the call carries an image AND the pipeline's
encode_prompt has an `image` parameter. img2img / inpaint also receive `image` (the
starting image) but their encode_prompt has none: they keep the cache, and the detailer its
gain.

Dummy pipelines; the signatures of the real diffusers classes are read without loading
a single weight.

Run:  .venv/Scripts/python tests/test_edit_prompt_image.py

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_pipeline as P


class _Pipe:
    """Counts the encodings and remembers what __call__ received."""

    _execution_device = "cpu"

    def __init__(self):
        self.text_encoder = object()
        self.encodes = []
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return type("o", (), {"images": ["IMG"]})()


class EditPipe(_Pipe):
    """encode_prompt takes the image, like QwenImageEdit(Plus)Pipeline."""

    def encode_prompt(self, prompt, image=None, device=None):
        self.encodes.append(image)
        return torch.zeros(1, 4, 8), torch.ones(1, 4)


class Img2ImgPipe(_Pipe):
    """encode_prompt with no image, like QwenImageImg2Img / QwenImageInpaint."""

    def encode_prompt(self, prompt, device=None):
        self.encodes.append(None)
        return torch.zeros(1, 4, 8), torch.ones(1, 4)


def _fresh():
    P._embed_cache_clear()
    P._EMBED_CACHE_MAX = 8                      # config.txt may have cut it down


def test_an_edit_is_encoded_with_its_image():
    _fresh()
    pipe = EditPipe()
    for _ in range(2):
        P._qwen_call(pipe, image="REF", prompt="make the car red")
    assert pipe.encodes == [], "the cache encoded the instruction without the image"
    for kw in pipe.calls:
        assert kw["prompt"] == "make the car red" and kw["image"] == "REF", kw
        assert not any(k in kw for k in P._EMBED_OUTS), "text-only embeddings were passed"
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
    """Their `image` is the starting image, not an input of the encoder."""
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
    """With no image, the encoding IS text-only: reusing it is right."""
    _fresh()
    pipe = EditPipe()
    P._qwen_call(pipe, prompt="p")
    P._qwen_call(pipe, prompt="p")
    assert pipe.encodes == [None], pipe.encodes
    print("OK test_without_an_image_the_edit_pipe_may_use_the_cache")


def test_the_installed_diffusers_signatures():
    """The rule rests on the signature: we check it on the real classes."""
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
