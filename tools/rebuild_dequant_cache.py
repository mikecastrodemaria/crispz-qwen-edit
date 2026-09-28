"""Pre-fills the dequantisation cache (cache/dequant) for ALL the FP8/INT8 checkpoints
of the model folders, so as not to pay for the conversion on the first use (it then
blocks the UI for several minutes in the middle of the work).

Usage:
    .venv/Scripts/python tools/rebuild_dequant_cache.py [--list] [--cpu]
    (or double-click on rebuild_cache.bat at the root)

- RESUMING IS FREE: a checkpoint already cached is skipped in a second -> re-runnable
  at will, including after an interruption.
- --list : shows what would be done, without converting anything.
- --cpu  : dequantises without touching the GPU (by default: the GPU when there is one,
  see convert_device). To be preferred when a render is running at the same time.

It concerns ONLY the FP8/INT8 .safetensors:
  - .gguf          -> stays quantised in VRAM, no dequantisation to cache;
  - bf16/fp16      -> nothing to dequantise (a cache would be a bf16 -> bf16 copy),
                      including in the ComfyUI layout where only the prefix is removed;
  - LoRA/SVDQuant  -> not loadable, skipped with their reason.

Every entry weighs as much as the complete BF16 build (~38 GB for a Qwen 20B
transformer): check that dequant_cache_max_gb (config.txt) covers the total, otherwise
the first conversions would be evicted by the last ones and the cache would be of no
use. Deleting cache/dequant is always safe (it rebuilds itself on demand).

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

# The size of an entry = the transformer's BF16 build (measured: 38.1 GiB for Qwen 20B).
ENTRY_GB = 38.0

if czp._dequant_cache_dir() is None:
    print("dequant_cache is 'off' in config.txt: nothing to pre-fill.")
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
            skipped.append((f, "bf16/fp16, nothing to dequantise"))
            continue
        cached = czp._dequant_cache_path(p)
        (done if cached and os.path.isfile(cached) else todo).append((p, dq))

for f, why in skipped:
    print(f"SKIP {f}: {why}")
for p, dq in done:
    print(f"ALREADY CACHED {os.path.basename(p)} ({dq})")
for p, dq in todo:
    print(f"TO CONVERT     {os.path.basename(p)} ({dq})")

if not todo and not done:
    print("\nNo FP8/INT8 checkpoint found in:", czp._checkpoint_dirs(),
          f"(--only {ONLY} filter)" if ONLY else "")
    sys.exit(0)

cap = czp.DEQUANT_CACHE_MAX_GB
need = (len(todo) + len(done)) * ENTRY_GB
print(f"\n{len(todo) + len(done)} checkpoint(s) to cover (~{need:.0f} GB of cache; "
      f"dequant_cache_max_gb ceiling = {cap:.0f} GB"
      + (", 0 = unlimited)" if cap <= 0 else ")"))
if 0 < cap < need:
    print(f"WARNING: a {cap:.0f} GB ceiling < the ~{need:.0f} GB needed -> the first "
          f"conversions would be evicted by the last ones and the cache would be of "
          f"no use.\nRaise dequant_cache_max_gb in config.txt "
          f"(>= {need:.0f}) before going on.")
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
        print(f"[{i}/{len(todo)}] OK {name} in {(time.time() - t0) / 60:.1f} min")
    except Exception as e:
        fail += 1
        print(f"[{i}/{len(todo)}] FAIL {name}: {type(e).__name__}: {e}")

print(f"\nDone in {(time.time() - t_all) / 60:.0f} min: {ok} converted, "
      f"{len(done)} already cached, {fail} failure(s).")
print("Re-runnable at will: everything already done is skipped.")
