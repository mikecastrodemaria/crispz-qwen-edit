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
git merge upstream/main      # seuls les conflits attendus sont dans cz_pipeline.py
```

`upstream` = crispz-studio (Z-Image). `origin` = ce fork.

## TODO

- [ ] Réécrire `cz_pipeline.py` pour Qwen-Image-Edit (branché sur `generate_omni`).
- [ ] Arbitrer édition-seule vs édition + txt2img.
- [ ] Valider l'offload VRAM (~20B) ; déclarer la limite dans le launcher si besoin.
- [ ] Adapter `requirements.txt` (transformers récent pour l'encodeur Qwen — déjà couvert).
- [ ] Mettre à jour README + identité.
- [ ] Launcher Pinokio `crispz-qwen-edit.pinokio.git` (clone ce repo).
