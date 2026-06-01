"""
make_new_gap_pipeline.py
========================
Reads gap_clicks.csv and produces:
  1.  HPC_scripts/Source_codes/diskdictionary_r90/diskdictionary_newgap.py
        – identical to diskdictionaryr0_5.py except rgap/wgap replaced with
          Huang+2018 half-depth values for every disk that was clicked.
  2.  HPC_scripts/inj_rev/new_gap/{disk}_gap{N}/
        – one directory per measured gap; scripts copied from inj_rev/r0_5/
          then patched (dict import, WORKDIR, target/gap_ix strings).

Run locally (Windows) before rsyncing to the HPC.
"""

import os, re, shutil, csv, sys
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
BASE         = r'D:\CPD_MPIA'
CSV_FILE     = rf'{BASE}\Injection_Recovery_Revised\gap_clicks.csv'
SRC_DICT     = rf'{BASE}\HPC_scripts\Source_codes\diskdictionary_r90\diskdictionaryr0_5.py'
NEW_DICT     = rf'{BASE}\HPC_scripts\Source_codes\diskdictionary_r90\diskdictionary_newgap.py'
OLD_PIPE     = rf'{BASE}\HPC_scripts\inj_rev\r0_5'
NEW_PIPE     = rf'{BASE}\Injection_Recovery_Revised'

# template to use for disks with no existing inj_rev/r0_5 directory (e.g. J1604, CQ_Tau)
TMPL_DISK, TMPL_IX = 'AA_Tau', 0

SKIP_EXT  = {'.log', '.fits', '.npz', '.txt', '.ipynb'}
SKIP_DIRS = {'resid_vis', 'mprofiles', 'recoveries', 'resid_images',
             'injections', 'assess_figs', '__pycache__'}

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Parse CSV → gaps_by_disk
# ═══════════════════════════════════════════════════════════════════════════════
gaps_by_disk = {}   # {disk: [{'inner_as', 'outer_as', 'gap_dip_as', 'method'}, ...]}

with open(CSV_FILE, newline='') as f:
    rows = list(csv.DictReader(f))

for disk in sorted({r['disk'] for r in rows}):
    disk_rows = [r for r in rows if r['disk'] == disk]
    sets, cur = [], {}
    for r in disk_rows:
        lbl = r['label']
        if lbl == 'gap_dip':
            if cur: sets.append(cur)
            cur = {'gap_dip_as': float(r['r_arcsec']), 'method': r.get('method', 'huang2018')}
        elif lbl == 'inner_edge':
            cur['inner_as'] = float(r['r_arcsec'])
        elif lbl == 'outer_edge':
            cur['outer_as'] = float(r['r_arcsec'])
    if cur: sets.append(cur)
    sets.sort(key=lambda s: s['gap_dip_as'])
    gaps_by_disk[disk] = sets

print("=== New gap parameters from CSV ===")
for disk, sets in gaps_by_disk.items():
    for i, gs in enumerate(sets):
        rgap = round((gs['inner_as'] + gs['outer_as']) / 2, 5)
        wgap = round(gs['outer_as'] - gs['inner_as'], 5)
        print(f"  {disk:15s} gap{i}:  rgap={rgap:6.4f}\"  wgap={wgap:6.4f}\"  [{gs['method']}]")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Write diskdictionary_newgap.py
# ═══════════════════════════════════════════════════════════════════════════════
with open(SRC_DICT) as f:
    dict_src = f.read()

# For each measured disk, replace 'rgap': [...] and 'wgap': [...] in the source text.
# Strategy: find the dict entry by 'name': 'DISK' then replace forward match.
for disk, sets in gaps_by_disk.items():
    new_rgap = [round((s['inner_as'] + s['outer_as']) / 2, 5) for s in sets]
    new_wgap = [round(s['outer_as'] - s['inner_as'], 5) for s in sets]

    for field, new_val in [('rgap', new_rgap), ('wgap', new_wgap)]:
        # match field inside this disk's entry (after 'name': 'DISK')
        pat = (r"('name':\s*'" + re.escape(disk) + r"'"
               r".*?'" + field + r"':\s*)\[[^\]]*\]")
        rep = r"\g<1>" + str(new_val)
        dict_src, n = re.subn(pat, rep, dict_src, count=1, flags=re.DOTALL)
        if n == 0:
            print(f"  WARNING: could not patch {disk}['{field}'] in dict source")

header = (
    "# diskdictionary_newgap.py\n"
    "# rgap / wgap updated from Huang+2018 half-depth measurements (gap_clicks.csv)\n"
    "# All other parameters unchanged from diskdictionaryr0_5.py\n\n"
)
with open(NEW_DICT, 'w') as f:
    f.write(header + dict_src)
print(f"\nWrote: {NEW_DICT}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Load old disk dictionary to find best-matching old gap indices
# ═══════════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(SRC_DICT))
import diskdictionaryr0_5 as _old_mod
old_dict = _old_mod.disk

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Build directory structure
# ═══════════════════════════════════════════════════════════════════════════════
os.makedirs(NEW_PIPE, exist_ok=True)
print("\n=== Creating pipeline directories ===")

def patch_file(fpath, src_disk, src_ix, dst_disk, dst_ix):
    """Apply all text substitutions to a single script file."""
    with open(fpath, encoding='utf-8') as f:
        txt = f.read()
    orig = txt

    # --- (a) combined disk+gap token (most specific → do first) ---
    txt = txt.replace(f'{src_disk}_gap{src_ix}', f'{dst_disk}_gap{dst_ix}')

    # --- (b) disk name in Python string literals ---
    if src_disk != dst_disk:
        txt = txt.replace(f"'{src_disk}'", f"'{dst_disk}'")
        txt = txt.replace(f'"{src_disk}"', f'"{dst_disk}"')

    # --- (c) gap index in Python assignments / string literals ---
    if src_ix != dst_ix:
        s, d = str(src_ix), str(dst_ix)
        # target, gap_ix, subsuf = 'DISK', 'N', '0'  (or double-quoted)
        for q in ('"', "'"):
            txt = txt.replace(
                f"target, gap_ix, subsuf = {q}{dst_disk}{q}, {q}{s}{q}",
                f"target, gap_ix, subsuf = {q}{dst_disk}{q}, {q}{d}{q}")
            txt = txt.replace(
                f", gap_ix, subsuf = {q}{dst_disk}{q}, {q}{s}{q}, {q}0{q}",
                f", gap_ix, subsuf = {q}{dst_disk}{q}, {q}{d}{q}, {q}0{q}")
        # standalone assignments
        txt = re.sub(r'\bgap_ix\s*=\s*' + re.escape(s) + r'\b',
                     f'gap_ix = {d}', txt)
        txt = re.sub(r'\bgap\s*=\s*' + re.escape(s) + r'\b',
                     f'gap = {d}', txt)
        txt = re.sub(r', gap_ix, subsuf = "' + re.escape(s) + r'"',
                     f', gap_ix, subsuf = "{d}"', txt)

    # --- (d) imageloop filename: {disk}_robust0_5_gap{ix} pattern ---
    txt = txt.replace(
        f'{src_disk}_robust0_5_gap{src_ix}',
        f'{dst_disk}_robust0_5_gap{dst_ix}')

    # --- (e) dict import ---
    txt = txt.replace('diskdictionaryr0_5', 'diskdictionary_newgap')

    # --- (f) WORKDIR in shell scripts (r0_5 → new_gap, after disk token replaced) ---
    txt = txt.replace(
        f'inj_rev/r0_5/{dst_disk}_gap{dst_ix}',
        f'inj_rev/new_gap/{dst_disk}_gap{dst_ix}')

    if txt != orig:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(txt)


for disk, sets in gaps_by_disk.items():
    for new_ix, gs in enumerate(sets):
        new_dir = os.path.join(NEW_PIPE, f'{disk}_gap{new_ix}')
        if os.path.isdir(new_dir):
            shutil.rmtree(new_dir)
        os.makedirs(new_dir)

        # --- find best-matching old source directory ---
        old_rgap = old_dict.get(disk, {}).get('rgap', [])
        rgap_mid = (gs['inner_as'] + gs['outer_as']) / 2
        if old_rgap:
            old_ix = int(np.argmin([abs(r - rgap_mid) for r in old_rgap]))
        else:
            old_ix = 0

        old_dir = os.path.join(OLD_PIPE, f'{disk}_gap{old_ix}')
        if os.path.isdir(old_dir):
            src_disk, src_ix = disk, old_ix
        else:
            # fall back to template disk
            old_dir = os.path.join(OLD_PIPE, f'{TMPL_DISK}_gap{TMPL_IX}')
            src_disk, src_ix = TMPL_DISK, TMPL_IX

        # --- copy files (skip generated output files) ---
        for fname in os.listdir(old_dir):
            fpath = os.path.join(old_dir, fname)
            if not os.path.isfile(fpath):
                continue
            _, ext = os.path.splitext(fname)
            if ext in SKIP_EXT:
                continue
            shutil.copy2(fpath, new_dir)

        # --- rename files: src_disk_gap{src_ix} → disk_gap{new_ix} ---
        for fname in sorted(os.listdir(new_dir)):
            new_fname = fname
            new_fname = new_fname.replace(f'{src_disk}_gap{src_ix}', f'{disk}_gap{new_ix}')
            if src_disk != disk:
                new_fname = new_fname.replace(src_disk, disk)
            if new_fname != fname:
                os.rename(os.path.join(new_dir, fname),
                          os.path.join(new_dir, new_fname))

        # --- patch content of every .py and .sh ---
        for fname in os.listdir(new_dir):
            fpath = os.path.join(new_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if fname.endswith('.py') or fname.endswith('.sh'):
                patch_file(fpath, src_disk, src_ix, disk, new_ix)

        rgap_r = round(rgap_mid, 4)
        wgap_r = round(gs['outer_as'] - gs['inner_as'], 4)
        print(f"  {disk}_gap{new_ix}  rgap={rgap_r:.4f}\"  wgap={wgap_r:.4f}\""
              f"  [{gs['method']:15s}]  <-- {src_disk}_gap{src_ix}")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
n_dirs = sum(len(v) for v in gaps_by_disk.values())
print(f"\nDone: {n_dirs} directories created under {NEW_PIPE}")
print("\nNext steps:")
print("  1. Inspect a few new_gap/ dirs and diskdictionary_newgap.py to verify patches")
print("  2. rsync Source_codes/diskdictionary_r90/diskdictionary_newgap.py to HPC")
print("  3. rsync HPC_scripts/inj_rev/new_gap/ to HPC inj_rev/new_gap/")
print("  4. Create SLURM submission scripts and launch")
