"""Pre-remplit le cache de dequantification (cache/dequant) pour TOUS les checkpoints
FP8/INT8 des dossiers de modeles, pour ne pas payer la conversion a la premiere
utilisation (elle bloque alors l'UI plusieurs minutes en plein travail).

Usage:
    .venv/Scripts/python tools/rebuild_dequant_cache.py [--list] [--cpu]
    (ou double-clic sur rebuild_cache.bat a la racine)

- REPRISE GRATUITE: un checkpoint deja en cache est saute en une seconde -> relancable
  a volonte, y compris apres une coupure.
- --list : montre ce qui serait fait, sans rien convertir.
- --cpu  : dequantification sans toucher au GPU (par defaut: GPU si present, cf.
  convert_device). A preferer si un rendu tourne en meme temps.

Ne concerne QUE les .safetensors FP8/INT8:
  - .gguf          -> reste quantifie en VRAM, aucune dequantification a cacher;
  - bf16/fp16      -> rien a dequantifier (un cache serait une copie bf16 -> bf16),
                      y compris au layout ComfyUI ou seul le prefixe est retire;
  - LoRA/SVDQuant  -> non chargeables, ignores avec leur raison.

Chaque entree pese autant que le build BF16 complet (~38 Go pour un transformer Qwen
20B): verifie que dequant_cache_max_gb (config.txt) couvre le total, sinon les
premieres conversions seraient evincees par les dernieres et le cache ne servirait
a rien. Supprimer cache/dequant est toujours sur (il se reconstruit a la demande).
"""
import os
import sys
import gc
import time

USAGE = """Usage: rebuild_dequant_cache.py [--list] [--cpu] [--only SUBSTR ...]
  --list          show what would be converted, convert nothing
  --cpu           dequantize on the CPU (GPU busy with a render)
  --only SUBSTR   only checkpoints whose file name contains SUBSTR (repeatable,
                  case-insensitive), e.g. --only jibMix --only 2511
Any other option (-h, --help, a typo) prints this and exits: the tool never
starts a multi-hour conversion by accident."""

_KNOWN = {"--list", "--cpu", "--only"}
ONLY = []
_args = sys.argv[1:]
_i = 0
while _i < len(_args):
    a = _args[_i]
    if a == "--only":
        if _i + 1 >= len(_args):
            print(USAGE)
            sys.exit(2)
        ONLY.append(_args[_i + 1].lower())
        _i += 2
        continue
    if a not in _KNOWN:
        print(USAGE)
        sys.exit(0 if a in ("-h", "--help") else 2)
    _i += 1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_pipeline as czp  # noqa: E402

if "--cpu" in sys.argv:
    czp.CONFIG["convert_device"] = "cpu"

# Taille d'une entree = le build BF16 du transformer (mesure: 38.1 Gio pour Qwen 20B).
ENTRY_GB = 38.0

if czp._dequant_cache_dir() is None:
    print("dequant_cache est sur 'off' dans config.txt: rien a pre-remplir.")
    sys.exit(0)

todo, done, skipped = [], [], []
for d in czp._checkpoint_dirs():
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if not os.path.isfile(p) or not f.lower().endswith(".safetensors"):
            continue
        if ONLY and not any(s in f.lower() for s in ONLY):
            continue
        bad = czp._safetensors_unsupported(p)
        if bad:
            skipped.append((f, bad))
            continue
        dq = czp._safetensors_dequant(p)
        if not dq:
            skipped.append((f, "bf16/fp16, rien a dequantifier"))
            continue
        cached = czp._dequant_cache_path(p)
        (done if cached and os.path.isfile(cached) else todo).append((p, dq))

for f, why in skipped:
    print(f"SKIP {f}: {why}")
for p, dq in done:
    print(f"DEJA EN CACHE {os.path.basename(p)} ({dq})")
for p, dq in todo:
    print(f"A CONVERTIR   {os.path.basename(p)} ({dq})")

if not todo and not done:
    print("\nAucun checkpoint FP8/INT8 trouve dans:", czp._checkpoint_dirs(),
          f"(filtre --only {ONLY})" if ONLY else "")
    sys.exit(0)

cap = czp.DEQUANT_CACHE_MAX_GB
need = (len(todo) + len(done)) * ENTRY_GB
print(f"\n{len(todo) + len(done)} checkpoint(s) a couvrir (~{need:.0f} Go de cache; "
      f"plafond dequant_cache_max_gb = {cap:.0f} Go"
      + (", 0 = illimite)" if cap <= 0 else ")"))
if 0 < cap < need:
    print(f"ATTENTION: plafond {cap:.0f} Go < ~{need:.0f} Go necessaires -> les "
          f"premieres conversions seraient evincees par les dernieres et le cache "
          f"ne servirait a rien.\nMonte dequant_cache_max_gb dans config.txt "
          f"(>= {need:.0f}) avant de continuer.")
    if "--list" not in sys.argv:
        sys.exit(1)

if "--list" in sys.argv:
    sys.exit(0)

t_all = time.time()
ok = fail = 0
for i, (p, dq) in enumerate(todo, 1):
    name = os.path.basename(p)
    t0 = time.time()
    print(f"\n[{i}/{len(todo)}] {name} ({dq}) ...")
    try:
        sd = czp._load_dequant_state_dict(p)
        czp._dequant_cache_store(p, sd)
        del sd
        gc.collect()
        ok += 1
        print(f"[{i}/{len(todo)}] OK {name} en {(time.time() - t0) / 60:.1f} min")
    except Exception as e:
        fail += 1
        print(f"[{i}/{len(todo)}] FAIL {name}: {type(e).__name__}: {e}")

print(f"\nTermine en {(time.time() - t_all) / 60:.0f} min: {ok} converti(s), "
      f"{len(done)} deja en cache, {fail} echec(s).")
print("Relancable a volonte: tout ce qui est fait est saute.")
