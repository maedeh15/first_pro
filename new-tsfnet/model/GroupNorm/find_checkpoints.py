"""
find_checkpoints.py -- run this to see what's ACTUALLY on disk, so we can fix
the path or filename pattern used by check_checkpoint_contamination.py.
Pure filesystem listing, no torch, instant, safe to run alongside a GPU job.
"""
import os
import glob

SEARCH_ROOT = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"
TARGET_HASH = "b5e27ee74d"

print(f"1) Does the results root exist?\n   {SEARCH_ROOT}\n   -> {os.path.isdir(SEARCH_ROOT)}\n")

if os.path.isdir(SEARCH_ROOT):
    print("2) Subfolders actually present under it:")
    for name in sorted(os.listdir(SEARCH_ROOT)):
        full = os.path.join(SEARCH_ROOT, name)
        if os.path.isdir(full):
            marker = "  <-- contains target hash" if TARGET_HASH in name else ""
            print(f"   {name}{marker}")

print(f"\n3) Any folder ANYWHERE under the results root containing '{TARGET_HASH}':")
matches = glob.glob(os.path.join(SEARCH_ROOT, f"*{TARGET_HASH}*"))
for m in matches:
    print(f"   {m}")

if matches:
    d = matches[0]
    print(f"\n4) Every file inside the first match ({d}):")
    for f in sorted(os.listdir(d))[:40]:
        print(f"   {f}")
    n_pt = len(glob.glob(os.path.join(d, "*.pt")))
    print(f"\n   total .pt files in this folder: {n_pt}")
else:
    print("\n   No folder with this hash found anywhere under the results root.")
    print("   Search the whole drive instead (slower):")
    print(f'   for root, dirs, files in os.walk(r"E:\\kntu"):')
    print(f'       if any("{TARGET_HASH}" in d for d in dirs):')
    print(f'           print(root)')
