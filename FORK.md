# crispz-qwen-edit — fork de crispz-studio

Ce dépôt est un **fork de [crispz-studio](https://github.com/mikecastrodemaria/crispz-studio)**
dont le moteur est remplacé par **Qwen-Image-Edit** (édition d'image par instruction).

- **Cible** : édition par prompt (« change le fond », « enlève cet objet »…) avec `QwenImageEditPipeline` (diffusers), ~20B.
- **Modèle HF** : `Qwen/Qwen-Image-Edit` (révision `Qwen-Image-Edit-2509` à arbitrer).

## Ce qui change vs l'upstream

Le SEUL fichier modèle à réécrire est **`cz_pipeline.py`** :

- Loader → `QwenImageEditPipeline.from_pretrained(..., torch_dtype=bfloat16)` (entrée = image + prompt d'édition).
- **Branchement principal sur `generate_omni`** : l'onglet « Omni/Edit » (image + référence + prompt) existe déjà dans `cz_ui.py` — seul le backend de chargement change.
- `txt2img_run` : soit délégué à `QwenImagePipeline` (génération pure), soit masqué si « édition seule ». À trancher.
- ~20B → **offload quasi obligatoire** sous 24 GB de VRAM ; défaut `OFFLOAD_MODE=model` côté launcher.
- LoRA → `load_lora_weights` (format Qwen-Image) si dispo.
- Refine/upscale : router via le pipeline edit (img2img-like) ou fallback ESRGAN seul (`cz_esrgan`) — à valider au prototypage.

Tout le reste (`cz_ui`, `cz_face`, `cz_esrgan`, `cz_ollama`, styles, CLI) est **conservé tel quel**.

## Workflow upstream (correctifs d'infra commune)

```
git fetch upstream
git merge upstream/main
```

`upstream` = crispz-studio (Z-Image). `origin` = ce fork.

**Conflits attendus : 4 fichiers** (pas seulement `cz_pipeline.py`). Règle de résolution :

| Fichier | Résolution | Delta fork à préserver |
|---|---|---|
| `cz_pipeline.py` | **ours** (couche modèle Qwen) + porter à la main les améliorations génériques d'upstream | tout le loader Qwen/GGUF, `_qwen_call`/`_cfg` |
| `cz_ui.py` | **theirs** (upstream) + réappliquer le delta fork | `PERFORMANCE_LORA`, `_set_performance` (pose aussi la LoRA du preset dans le slot 1), pré-remplissage des slots depuis `default_loras`, défaut du dropdown offload, `ZIMAGE_BASE_REPOS`, libellés « Qwen … » |
| `cz_core.py` | **theirs** + réappliquer | `DEFAULT_BASE_REPO`, `MODEL_PROFILES`/`DEFAULT_MODEL_PROFILE`, `.gguf` dans `_is_single_file` |
| `config-sample.txt` | **theirs** + réappliquer | presets/profils Qwen, `default_*`, `default_loras`, `zimage_omni_base` |

### Ce qui se porte depuis upstream vers `cz_pipeline.py` (générique)

`_load_monitor`/`_fmt_load`/`_load_pct` (progression de chargement), `_apply_loras` +
`_APPLIED_LORAS` (hot-swap LoRA PEFT — `set_loras` ne doit **pas** appeler `free_vram`),
`_lora_weight_range` + `LORA_WEIGHT_MIN/MAX`, `_swap_transformer`/`_load_transformer`
(recharger le transformer seul), `_LAST_SEED`/`_NO_SEED_INCREMENT`/`_SAVE_PRE_UPSCALE`
+ leurs setters (requis par `cz_ui`).

### Ce qui NE se porte PAS (spécifique Z-Image)

`round_to_multiple(x, m=32)` → **Qwen reste en `m=16`** (les callsites qui exigent /32 le
passent explicitement). Idem le snap /32 de `_refine_whole`, et tout import
`ZImagePipeline`/`ZImageTransformer2DModel`.

### Après merge, vérifier

Contrat d'API (`cz_ui`/`cz_cli` importent tout ce que `cz_pipeline` doit exposer),
`cz_ui.build_ui()` headless, `round_to_multiple(100) == 96`, defaults Qwen intacts,
`PERFORMANCE_LORA` bien parsé, `tests/` (surtout `test_lora_hotswap`, `test_load`,
`test_model_swap`).

## État

Architecture retenue : **studio complet + onglet Edit** (les deux décidés).
- base txt2img/img2img/inpaint = **Qwen-Image** (`Qwen/Qwen-Image`)
- onglet Omni/Edit = **Qwen-Image-Edit-2509** (modèle séparé, multi-images), via `generate_omni`.

- [x] Réécrire `cz_pipeline.py` : `QwenImagePipeline` / `QwenImageImg2ImgPipeline` /
      `QwenImageInpaintPipeline` (base) + `QwenImageEditPlusPipeline` (onglet Edit, repli
      `QwenImageEditPipeline`). Tous les appels passent par `_qwen_call` + `_cfg`
      (`true_cfg_scale` = curseur guidance, `guidance_scale` distillé = 1.0, negative actif).
- [x] `generate_omni` rebranché sur l'édition par instruction (image(s) + prompt), dimensions
      d'entrée préservées. API publique du module **inchangée** (cz_ui/cz_cli intacts).
- [x] Défauts Qwen : `config-sample.txt` (true_cfg 4.0, 30 steps, presets, profils qwen/edit,
      `zimage_omni_model = Qwen/Qwen-Image-Edit-2509`) + `cz_core.py`.
- [x] Vérifié : `import app` OK, contrat d'API complet préservé, classes Qwen présentes,
      defaults corrects (base=Qwen-Image, edit=2509, cfg=4.0, profil=(30,4.0)).
- [ ] `requirements.txt` : Qwen-Image/Edit déjà dans diffusers (source) + transformers récent
      (déjà couvert). À confirmer au 1er run.
- [ ] Valider l'offload VRAM (~20B base + ~20B edit) ; défaut `OFFLOAD_MODE=model` côté launcher.
- [ ] Test génération + édition réels sur GPU (download des deux modèles).
- [ ] Mettre à jour README + identité (titres, captures).
- [ ] Launcher Pinokio `crispz-qwen-edit.pinokio.git` (clone ce repo, `OFFLOAD_MODE=model`).
