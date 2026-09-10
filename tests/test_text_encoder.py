"""Encodeur texte de remplacement (Models > Checkpoints > Text encoder).

Qwen-Image lit le DERNIER etat cache de l'encodeur Qwen2.5-VL-7B dans txt_norm +
txt_in, larges de 3584: un autre encodeur ne se branche que s'il a la meme famille
(qwen2_5_vl -- le pipe d'edition lui montre l'image), la meme largeur et le meme
nombre de couches. Un Qwen2.5-VL-7B "abliterated" convient; un 3B (2048 de large) ne
peut pas marcher, et le refus doit le dire AVANT de lire 16 Go.

L'option vaut pour les DEUX pipes: le base (QwenImagePipeline, partage par img2img et
inpaint via from_pipe) et le pipe d'edition (_load_omni), compare a SON propre repo.

Ces tests verrouillent aussi ce qui rendrait l'option dangereuse en silence:
  - un changement d'encodeur libere les deux pipes et vide le cache d'embeddings, et
    l'encodeur fait partie de la CLE du cache -- id(enc) seul ne suffit pas, CPython
    recycle les id d'objets liberes;
  - un encodeur qui ne convient pas, ou qui plante au chargement, est ecarte: le pipe
    tourne avec celui de son repo, la generation ne tombe jamais;
  - les metadonnees nomment l'encodeur qui a REELLEMENT tourne (celui du pipe
    d'edition pour une edition), par son nom de dossier et jamais par son chemin;
  - la file garde l'encodeur du job.

Aucun modele n'est charge: configs ecrites en dossier temporaire, pipelines factices,
lecteur de config du repo de base remplace (pas de reseau).

Run:  .venv/Scripts/python tests/test_text_encoder.py
"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import cz_imageio
import cz_pipeline as P

# Qwen2.5-VL-7B tel que range dans Qwen/Qwen-Image/text_encoder: largeur et couches au
# premier niveau ET sous text_config (transformers >= 4.53).
QWEN_VL_7B = {"model_type": "qwen2_5_vl", "hidden_size": 3584, "num_hidden_layers": 28,
              "architectures": ["Qwen2_5_VLForConditionalGeneration"],
              "text_config": {"model_type": "qwen2_5_vl_text", "hidden_size": 3584,
                              "num_hidden_layers": 28}}
QWEN_VL_3B = {"model_type": "qwen2_5_vl", "hidden_size": 2048, "num_hidden_layers": 36,
              "architectures": ["Qwen2_5_VLForConditionalGeneration"]}


def _folder(cfg, sub=None, name="enc"):
    root = tempfile.mkdtemp(prefix="te_")
    d = os.path.join(root, name)
    p = os.path.join(d, sub) if sub else d
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return d


class _Base:
    """Remplace la config de l'encodeur du repo de base (pas de reseau, pas de HF) et
    retient pour quel repo elle a ete demandee."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.asked = []

    def __enter__(self):
        self.old = P._base_text_encoder_config

        def fake(base=None):
            self.asked.append(base)
            return self.cfg
        P._base_text_encoder_config = fake
        return self

    def __exit__(self, *a):
        P._base_text_encoder_config = self.old


class _Patch:
    """Pose des attributs le temps d'un test puis remet les anciens (ou les retire
    s'ils n'existaient pas: modules diffusers / huggingface_hub a import paresseux)."""

    def __init__(self, *triples):
        self.triples = triples

    def __enter__(self):
        self.saved = []
        for obj, name, val in self.triples:
            d = vars(obj)
            self.saved.append((obj, name, name in d, d.get(name)))
            setattr(obj, name, val)
        return self

    def __exit__(self, *a):
        for obj, name, had, val in reversed(self.saved):
            if had:
                setattr(obj, name, val)
            else:
                delattr(obj, name)


class _FakePipe:
    """Pipeline diffusers factice: retient ce que from_pretrained / le constructeur ont
    recu. scheduler=None: _ensure_base n'a aucune config de scheduler a capturer."""

    def __init__(self, **kw):
        self.kw = kw
        self.repo = None
        self.scheduler = None

    @classmethod
    def from_pretrained(cls, repo, **kw):
        p = cls(**kw)
        p.repo = repo
        return p

    def to(self, *a, **k):
        return self


def _boom(src, base=None):
    raise OSError("safetensors truncated")


def test_same_architecture_is_accepted():
    with _Base(QWEN_VL_7B):
        assert P._text_encoder_problem(_folder(QWEN_VL_7B)) is None
        # poids dans un sous-dossier text_encoder/ (copie d'un repo diffusers)
        assert P._text_encoder_problem(_folder(QWEN_VL_7B, "text_encoder")) is None
        # config recente: la partie texte seulement sous text_config
        nested = {"model_type": "qwen2_5_vl",
                  "text_config": {"hidden_size": 3584, "num_hidden_layers": 28}}
        assert P._text_encoder_problem(_folder(nested)) is None
    print("OK test_same_architecture_is_accepted")


def test_the_text_config_is_what_counts():
    assert P._enc_dims(QWEN_VL_7B) == (3584, 28, "qwen2_5_vl")
    lying_top = {"model_type": "qwen2_5_vl", "hidden_size": 3584, "num_hidden_layers": 28,
                 "text_config": {"hidden_size": 2048, "num_hidden_layers": 36}}
    assert P._enc_dims(lying_top) == (2048, 36, "qwen2_5_vl")
    with _Base(QWEN_VL_7B):
        why = P._text_encoder_problem(_folder(lying_top))
    assert why and "2048" in why and "3584" in why, why
    print("OK test_the_text_config_is_what_counts")


def test_a_narrower_encoder_is_refused_with_both_widths():
    with _Base(QWEN_VL_7B):
        why = P._text_encoder_problem(_folder(QWEN_VL_3B))
    assert why and "2048" in why and "3584" in why, why
    assert "FLUX" not in why, why                     # pas d'indice herite de klein
    print("OK test_a_narrower_encoder_is_refused_with_both_widths")


def test_other_family_and_layer_count_are_refused():
    with _Base(QWEN_VL_7B):
        # Qwen2.5-7B texte seul: meme largeur, meme profondeur, mais aucune vision
        why = P._text_encoder_problem(_folder({"model_type": "qwen2", "hidden_size": 3584,
                                               "num_hidden_layers": 28}))
        assert why and "'qwen2'" in why and "'qwen2_5_vl'" in why, why
        deeper = {**QWEN_VL_7B, "num_hidden_layers": 36,
                  "text_config": {"hidden_size": 3584, "num_hidden_layers": 36}}
        why = P._text_encoder_problem(_folder(deeper))
        assert why and "36" in why and "28" in why, why
    print("OK test_other_family_and_layer_count_are_refused")


def test_gguf_single_file_and_empty_folder_are_refused_with_the_reason():
    with _Base(QWEN_VL_7B):
        assert "GGUF" in P._text_encoder_problem(r"F:\x\Qwen2.5-VL-7B-Instruct-Q8_0.gguf")
        assert "FOLDER" in P._text_encoder_problem(r"F:\x\qwen_2.5_vl_7b_fp8_scaled.safetensors")
        assert "config.json" in P._text_encoder_problem(tempfile.mkdtemp())
    print("OK test_gguf_single_file_and_empty_folder_are_refused_with_the_reason")


def test_hf_ids_may_carry_a_subfolder():
    assert P._split_hf_src("owner/repo") == ("owner/repo", None)
    assert P._split_hf_src("owner/repo/text_encoder") == ("owner/repo", "text_encoder")
    assert P._split_hf_src("owner/repo/a/b") == ("owner/repo", "a/b")
    print("OK test_hf_ids_may_carry_a_subfolder")


def test_the_class_comes_from_the_base_repo_model_index():
    base = tempfile.mkdtemp(prefix="base_")
    with open(os.path.join(base, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump({"text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"]}, f)
    cls = P._encoder_class(base)
    assert cls.__name__ == "Qwen2_5_VLForConditionalGeneration", cls
    print("OK test_the_class_comes_from_the_base_repo_model_index")


def test_changing_the_encoder_frees_both_pipes_and_the_cache():
    old = (P.TEXT_ENCODER, P._BASE_PIPE, P._DERIVED,
           P._TEXT_ENCODER_ACTIVE, P._TEXT_ENCODER_ACTIVE_EDIT)
    try:
        P.TEXT_ENCODER = ""
        P._BASE_PIPE = object()
        P._DERIVED = {"txt2img": P._BASE_PIPE, "omni": object()}
        P._TEXT_ENCODER_ACTIVE = P._TEXT_ENCODER_ACTIVE_EDIT = r"D:\enc\old"
        P._EMBED_CACHE[("k",)] = ("v",)
        P.set_text_encoder(r"D:\enc\qwen25vl-abl")
        assert P.TEXT_ENCODER == r"D:\enc\qwen25vl-abl"
        assert P._BASE_PIPE is None, "le pipeline doit etre libere"
        assert "omni" not in P._DERIVED, "le pipe d'edition garderait l'ancien encodeur"
        assert not P._EMBED_CACHE, "les anciens encodages resteraient servis"
        assert P._TEXT_ENCODER_ACTIVE == "" and P._TEXT_ENCODER_ACTIVE_EDIT == ""
        # meme valeur: rien ne bouge, pas de rechargement inutile
        sentinel = P._BASE_PIPE = object()
        P.set_text_encoder(r"D:\enc\qwen25vl-abl")
        assert P._BASE_PIPE is sentinel
    finally:
        (P.TEXT_ENCODER, P._BASE_PIPE, P._DERIVED,
         P._TEXT_ENCODER_ACTIVE, P._TEXT_ENCODER_ACTIVE_EDIT) = old
        P._EMBED_CACHE.clear()
    print("OK test_changing_the_encoder_frees_both_pipes_and_the_cache")


class _EncPipe:
    def __init__(self):
        self.text_encoder = object()
        self._execution_device = "cpu"
        self.n = 0

    def encode_prompt(self, prompt, device=None):
        self.n += 1
        return (torch.zeros(1, 2, 4), torch.ones(1, 2))


def test_the_embed_key_carries_the_encoder():
    """Meme prompt, meme objet pipe, deux encodeurs: deux encodages."""
    P._embed_cache_clear()
    old = (P._TEXT_ENCODER_ACTIVE, P._EMBED_CACHE_MAX)
    try:
        P._EMBED_CACHE_MAX = 8                 # config.txt peut l'avoir coupe
        pipe = _EncPipe()
        P._TEXT_ENCODER_ACTIVE = ""
        P._cached_prompt_embeds(pipe, "p", {})
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 1, pipe.n
        P._TEXT_ENCODER_ACTIVE = r"D:\enc\qwen25vl-abl"
        P._cached_prompt_embeds(pipe, "p", {})
        assert pipe.n == 2, "un encodage de l'autre encodeur a ete resservi"
    finally:
        P._TEXT_ENCODER_ACTIVE, P._EMBED_CACHE_MAX = old
        P._embed_cache_clear()
    print("OK test_the_embed_key_carries_the_encoder")


def test_metadata_names_the_encoder_that_ran_and_never_its_path():
    old = (P.TEXT_ENCODER, P._TEXT_ENCODER_ACTIVE, P._TEXT_ENCODER_ACTIVE_EDIT)
    path = r"C:\Users\someone\models\text_encoders\qwen25vl-7b-abliterated"
    name = "qwen25vl-7b-abliterated"
    try:
        P.TEXT_ENCODER = P._TEXT_ENCODER_ACTIVE = path
        P._TEXT_ENCODER_ACTIVE_EDIT = ""
        m = P._gen_meta("txt2img", "p")
        assert m["text_encoder"] == name, m
        assert "someone" not in json.dumps(m), "chemin local dans les metadonnees"
        # une edition vient du pipe d'edition, charge a part: lui l'a ecarte
        m = P._gen_meta("omni", "p")
        assert "text_encoder" not in m and m["text_encoder_not_applied"] == name, m
        # l'inverse: onglet Edit seul, le base n'a jamais tourne avec
        P._TEXT_ENCODER_ACTIVE, P._TEXT_ENCODER_ACTIVE_EDIT = "", path
        assert P._gen_meta("omni", "p")["text_encoder"] == name
        m = P._gen_meta("txt2img", "p")
        assert "text_encoder" not in m and m["text_encoder_not_applied"] == name, m
        # rien de demande: rien d'ecrit
        P.TEXT_ENCODER = P._TEXT_ENCODER_ACTIVE_EDIT = ""
        for mode in ("txt2img", "omni"):
            m = P._gen_meta(mode, "p")
            assert "text_encoder" not in m and "text_encoder_not_applied" not in m, m
    finally:
        P.TEXT_ENCODER, P._TEXT_ENCODER_ACTIVE, P._TEXT_ENCODER_ACTIVE_EDIT = old
    assert P._encoder_label(r"D:\m\qwen25vl-abl\text_encoder") == "qwen25vl-abl"
    assert P._encoder_label("owner/repo/sub") == "owner/repo/sub"
    line = cz_imageio._a1111_parameters({"prompt": "p", "text_encoder": name})
    assert f"Text encoder: {name}" in line, line
    print("OK test_metadata_names_the_encoder_that_ran_and_never_its_path")


def test_the_list_finds_encoder_folders():
    d = _folder(QWEN_VL_7B, name="qwen25vl-7b-abliterated")
    root = os.path.dirname(d)
    nested = os.path.join(root, "diffusers-copy")
    os.makedirs(os.path.join(nested, "text_encoder"))
    with open(os.path.join(nested, "text_encoder", "config.json"), "w", encoding="utf-8") as f:
        json.dump(QWEN_VL_7B, f)
    os.makedirs(os.path.join(root, "empty"))
    old = P.TEXT_ENCODERS_DIR
    try:
        P.TEXT_ENCODERS_DIR = root
        found = P.list_text_encoders()
    finally:
        P.TEXT_ENCODERS_DIR = old
    assert d in found and nested in found, found
    assert not any(f.endswith("empty") for f in found), found
    print("OK test_the_list_finds_encoder_folders")


def test_the_base_pipe_gets_the_encoder_or_falls_back():
    import diffusers
    enc = _folder(QWEN_VL_7B, name="qwen25vl-abl")
    loads = []

    def fake_load(src, base=None):
        loads.append((src, base))
        return "ENC"
    P.free_vram()
    try:
        # DEVICE force a cpu: sous Windows, CUDA_VISIBLE_DEVICES vide ne cache pas le GPU
        # (-1 le fait) -- ce test ne doit prendre aucun chemin CUDA quoi qu'il arrive.
        with _Patch((diffusers, "QwenImagePipeline", _FakePipe), (P, "DEVICE", "cpu"),
                    (P, "BASE_REPO", "fake/qwen-image"), (P, "ZIMAGE_TRANSFORMER", None),
                    (P, "LORAS", []), (P, "_BASE_SCHED_CONFIG", None),
                    (P, "_apply_sampler", lambda p: None),
                    (P, "_load_text_encoder", fake_load),
                    (P, "TEXT_ENCODER", enc)), _Base(QWEN_VL_7B) as base:
            pipe = P._ensure_base()
            assert pipe.repo == "fake/qwen-image" and pipe.kw.get("text_encoder") == "ENC", pipe.kw
            assert P._TEXT_ENCODER_ACTIVE == enc
            assert base.asked == ["fake/qwen-image"], base.asked
            assert loads == [(enc, "fake/qwen-image")], loads
            # ne convient plus au repo: ecarte, l'encodeur du repo tourne
            P.free_vram()
            base.cfg = QWEN_VL_3B
            pipe = P._ensure_base()
            assert "text_encoder" not in pipe.kw and P._TEXT_ENCODER_ACTIVE == "", pipe.kw
            # plante au chargement: meme repli, la generation ne tombe pas
            P.free_vram()
            base.cfg = QWEN_VL_7B
            P._load_text_encoder = _boom
            pipe = P._ensure_base()
            assert "text_encoder" not in pipe.kw and P._TEXT_ENCODER_ACTIVE == "", pipe.kw
    finally:
        P.free_vram()
    print("OK test_the_base_pipe_gets_the_encoder_or_falls_back")


def test_the_edit_pipe_gets_it_too_checked_against_its_own_repo():
    import diffusers
    import huggingface_hub
    enc = _folder(QWEN_VL_7B, name="qwen25vl-abl")
    tmp = tempfile.mkdtemp(prefix="omni_")
    gguf = os.path.join(tmp, "qwen-image-edit-2511-Q4_K_M.gguf")
    open(gguf, "wb").close()
    # composants du repo d'edition: une classe factice qui note ce qu'on lui demande
    fetched = []

    class Comp:
        @classmethod
        def from_pretrained(cls, repo, subfolder=None, **kw):
            fetched.append(subfolder)
            return f"{subfolder}@{repo}"
    mod = types.ModuleType("_cz_fake_components")
    mod.Comp = Comp
    sys.modules[mod.__name__] = mod
    mi = os.path.join(tmp, "model_index.json")
    with open(mi, "w", encoding="utf-8") as f:
        json.dump({n: [mod.__name__, "Comp"]
                   for n in ("scheduler", "vae", "text_encoder", "tokenizer", "processor")}, f)
    loads = []

    def fake_load(src, base=None):
        loads.append(base)
        return "ENC"
    old_env = os.environ.get("QWEN_EDIT_BASE")
    os.environ["QWEN_EDIT_BASE"] = "fake/edit-base"
    P.free_vram()
    try:
        with _Patch((diffusers, "QwenImageEditPlusPipeline", _FakePipe), (P, "DEVICE", "cpu"),
                    (huggingface_hub, "hf_hub_download", lambda repo, fn, **k: mi),
                    (P, "_load_transformer", lambda path=None, base=None: "TF"),
                    (P, "_load_text_encoder", fake_load),
                    (P, "OMNI_MODEL", gguf), (P, "TEXT_ENCODER", enc)), _Base(QWEN_VL_7B) as base:
            # 1) transformer single-file: pipe monte composant par composant
            pipe = P._load_omni()
            assert pipe.kw["text_encoder"] == "ENC", pipe.kw
            assert "text_encoder" not in fetched, "l'encodeur du repo aurait ete lu pour rien"
            assert pipe.kw["tokenizer"] == "tokenizer@fake/edit-base", pipe.kw
            assert pipe.kw["processor"] == "processor@fake/edit-base", pipe.kw
            assert base.asked == ["fake/edit-base"] and loads == ["fake/edit-base"]
            assert P._TEXT_ENCODER_ACTIVE_EDIT == enc and P._TEXT_ENCODER_ACTIVE == ""
            # ne convient pas au repo d'edition: il garde le sien
            P.free_vram()
            fetched.clear()
            base.cfg = QWEN_VL_3B
            pipe = P._load_omni()
            assert pipe.kw["text_encoder"] == "text_encoder@fake/edit-base", pipe.kw
            assert P._TEXT_ENCODER_ACTIVE_EDIT == ""
            # 2) repo d'edition complet: text_encoder= dans from_pretrained, compare a CE repo
            P.free_vram()
            base.cfg, base.asked[:], loads[:] = QWEN_VL_7B, [], []
            P.OMNI_MODEL = "fake/Qwen-Image-Edit-2509"
            pipe = P._load_omni()
            assert pipe.repo == "fake/Qwen-Image-Edit-2509", pipe.repo
            assert pipe.kw.get("text_encoder") == "ENC", pipe.kw
            assert base.asked == ["fake/Qwen-Image-Edit-2509"], base.asked
            assert P._TEXT_ENCODER_ACTIVE_EDIT == enc
            # plante au chargement: le repo complet charge son propre encodeur
            P.free_vram()
            P._load_text_encoder = _boom
            pipe = P._load_omni()
            assert "text_encoder" not in pipe.kw and P._TEXT_ENCODER_ACTIVE_EDIT == "", pipe.kw
    finally:
        if old_env is None:
            os.environ.pop("QWEN_EDIT_BASE", None)
        else:
            os.environ["QWEN_EDIT_BASE"] = old_env
        sys.modules.pop(mod.__name__, None)
        P.free_vram()
    print("OK test_the_edit_pipe_gets_it_too_checked_against_its_own_repo")


def test_the_ui_saves_only_a_valid_choice():
    import cz_ui as U
    saves, calls = [], []
    good = _folder(QWEN_VL_7B, name="qwen25vl-abl")
    bad = _folder(QWEN_VL_3B, name="qwen25vl-3b")
    with _Patch((U, "_save_prefs_keys", saves.append),
                (P, "set_text_encoder", calls.append),
                (P, "TEXT_ENCODER", "")), _Base(QWEN_VL_7B):
        msg = U._ui_set_text_encoder(bad)
        assert "not applied" in msg and "2048" in msg and "3584" in msg, msg
        assert saves == [] and calls == [], (saves, calls)
        msg = U._ui_set_text_encoder(good)
        assert "qwen25vl-abl" in msg, msg
        assert calls == [good] and saves == [{"text_encoder": good}], (saves, calls)
        calls.clear()
        saves.clear()
        U._ui_set_text_encoder("")
        assert calls == [""] and saves == [{"text_encoder": ""}], (saves, calls)
        # une valeur collee (repo HF) reste proposee dans le dropdown
        P.TEXT_ENCODER = "owner/qwen25vl-abl"
        assert ("owner/qwen25vl-abl", "owner/qwen25vl-abl") in U._te_choices()
    print("OK test_the_ui_saves_only_a_valid_choice")


def test_the_queue_keeps_the_encoder():
    import cz_ui as U
    calls = []
    old = (P.TEXT_ENCODER, P.set_text_encoder)
    try:
        P.TEXT_ENCODER = r"D:\enc\qwen25vl-abl"
        ms = U._q_model_state()
        assert ms["text_encoder"] == r"D:\enc\qwen25vl-abl", ms
        P.set_text_encoder = lambda s: calls.append(s)
        U._q_restore_model_state(ms)
        assert calls == [r"D:\enc\qwen25vl-abl"], calls
        # snapshot d'avant l'option: on ne touche pas a l'encodeur courant
        calls.clear()
        U._q_restore_model_state({k: v for k, v in ms.items() if k != "text_encoder"})
        assert calls == [], calls
    finally:
        P.TEXT_ENCODER, P.set_text_encoder = old
    print("OK test_the_queue_keeps_the_encoder")



def test_default_picked_in_the_ui_survives_a_restart():
    """Choisir "Default" ecrit "" dans les preferences: au redemarrage, une valeur de
    config.txt ne doit pas revenir par-dessus. L'environnement gagne toujours."""
    cfg = {"text_encoder": r"D:\enc\from-config"}
    assert P._resolve_text_encoder({}, {}, cfg) == r"D:\enc\from-config"
    assert P._resolve_text_encoder({}, {"text_encoder": ""}, cfg) == ""
    assert P._resolve_text_encoder({}, {"text_encoder": r"D:\enc\ui"}, cfg) == r"D:\enc\ui"
    assert P._resolve_text_encoder({"QWEN_TEXT_ENCODER": r"D:\enc\env"},
                                   {"text_encoder": ""}, cfg) == r"D:\enc\env"
    print("OK test_default_picked_in_the_ui_survives_a_restart")


def test_compatible_encoders_in_the_hf_cache_are_listed():
    """Un encodeur telecharge depuis HF vit dans le cache HF: la liste doit le montrer.
    Pas un pipeline diffusers, pas une config sans poids; une autre taille est nommee a cote."""
    import json as _json
    import os as _os
    import tempfile as _tempfile
    ref = {"model_type": "fam", "hidden_size": 64, "num_hidden_layers": 2}
    wide = {"model_type": "fam", "hidden_size": 128, "num_hidden_layers": 2}
    root = _tempfile.mkdtemp(prefix="hfcache_")

    def snap(repo, sub=None, cfg=ref, weights=True, pipeline=False):
        d = _os.path.join(root, "models--" + repo.replace("/", "--"), "snapshots", "r1")
        p = _os.path.join(d, sub) if sub else d
        _os.makedirs(p, exist_ok=True)
        with open(_os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
            _json.dump(cfg, f)
        if weights:
            open(_os.path.join(p, "model.safetensors"), "wb").close()
        if pipeline:
            with open(_os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
                f.write("{}")

    snap("a/fits")
    snap("b/fits-in-sub", sub="enc")
    snap("c/wider", cfg=wide)
    snap("d/pipeline", sub="text_encoder", pipeline=True)
    snap("e/config-only", weights=False)
    snap("f/no-shape", cfg={"_class_name": "AutoencoderKL"})
    old = (P._hf_cache_dir, P._base_text_encoder_config)
    try:
        P._hf_cache_dir = lambda: root
        P._base_text_encoder_config = lambda base=None: ref
        got = [v for _l, v in P.list_cached_text_encoders()]
        other, width = P.cached_text_encoder_mismatches()
        import cz_ui as U
        hint = U._te_hint()
        choices = [v for _l, v in U._te_choices()]
    finally:
        P._hf_cache_dir, P._base_text_encoder_config = old
    assert got == ["a/fits", "b/fits-in-sub/enc"], got
    assert all(v in choices for v in got), choices
    assert [h for h, _w in other] == ["c/wider"] and width == 64, (other, width)
    assert "128" in hint and "64" in hint and "c/wider" in hint, hint
    print("OK test_compatible_encoders_in_the_hf_cache_are_listed")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All text-encoder tests passed.")
