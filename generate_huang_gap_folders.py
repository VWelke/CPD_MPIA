#!/usr/bin/env python3
"""
generate_huang_gap_folders.py
Creates HPC_huang_gap/{gap,inner_cavity}/{disk}_gap{ix}/ folders with all
pipeline scripts and assess_recovery.ipynb, adapted for diskdictionary_newgap.
Run from D:\CPD_MPIA (or any directory with correct absolute paths below).
"""

import os, shutil, json

BASE_DIR        = r'd:\CPD_MPIA\HPC_huang_gap'
EQC1_TMPL_DIR   = r'd:\CPD_MPIA\HPC_eqC1_gap\gap\AA_Tau_gap0'
NB_TMPL_PATH    = os.path.join(EQC1_TMPL_DIR, 'assess_recovery.ipynb')

# (disk, gap_ix, subfolder)  — same classification as eqC1
GAPS = [
    ('AA_Tau',      0, 'gap'),
    ('AA_Tau',      1, 'gap'),
    ('CQ_Tau',      0, 'inner_cavity'),
    ('DM_Tau',      0, 'inner_cavity'),
    ('HD_135344B',  0, 'gap'),
    ('HD_135344B',  1, 'inner_cavity'),
    ('J1604',       0, 'inner_cavity'),
    ('J1615',       0, 'gap'),
    ('J1615',       1, 'inner_cavity'),
    ('J1852',       0, 'inner_cavity'),
    ('LkCa_15',     0, 'gap'),
    ('LkCa_15',     1, 'inner_cavity'),
    ('MWC_758',     0, 'inner_cavity'),
    ('MWC_758',     1, 'gap'),
    ('SY_Cha',      0, 'inner_cavity'),
    ('V4046_Sgr',   0, 'inner_cavity'),
    ('V4046_Sgr',   1, 'gap'),
]

# ─────────────────────────────────────────────────────────────────────────────
def read_tmpl(fname):
    with open(os.path.join(EQC1_TMPL_DIR, fname), 'r', encoding='utf-8') as f:
        return f.read()

def write_lf(path, content):
    """Write with Unix (LF) line endings regardless of platform."""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)

# ─────────────────────────────────────────────────────────────────────────────
# Load notebook template as raw text (we use plain string replacements)
with open(NB_TMPL_PATH, 'r', encoding='utf-8') as f:
    NB_TEMPLATE = f.read()

# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh template (avoids f-string brace collisions with bash vars)
SH_TEMPLATE = """\
#!/usr/bin/env bash

WORKDIR="/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/SUBFOLDER/DISK_GAP"
VENV="/nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate"
CASA_BIN="/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/bin"
export CASA_CONFIG="/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py"

cd "$WORKDIR"
mkdir -p resid_vis mprofiles recoveries resid_images injections
source "$VENV"
export PATH="${CASA_BIN}:$PATH"

if [ -f "injections/DISK_GAP_mpars.0.txt" ]; then
    echo "=== [DISK_GAP] Inject already done, skipping ==="
else
    echo "=== [DISK_GAP] Inject start: $(date) ==="
    OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=2 \\
    OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
    python -u DISK_GAP_injectloop.py > inject.log 2>&1
    echo "=== Inject finished: $(date) ===" >> inject.log
fi

if compgen -G "*.custom.mask" > /dev/null 2>&1; then
    echo "=== Prepimaging already done, skipping ==="
else
    echo "=== [DISK_GAP] Prepimaging start: $(date) ==="
    CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \\
    MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
    python -u prepimaging.py > prepimaging.log 2>&1
    echo "=== Prepimaging finished: $(date) ===" >> prepimaging.log
fi

echo "=== [DISK_GAP] Imageloop start: $(date) ==="
CASA_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE \\
MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
casa --pipeline --configfile "$CASA_CONFIG" --nogui --nologger --nologfile \\
  -c DISK_robust0_5_GAPIX_imageloop.py > imageloop.log 2>&1
echo "=== Imageloop finished: $(date) ===" >> imageloop.log

echo "=== [DISK_GAP] Recovery start: $(date) ==="
python -u recover_loop.py > recover_loop.log 2>&1
echo "=== Recovery finished: $(date) ===" >> recover_loop.log

deactivate

cd "$WORKDIR/resid_images"
cp $(ls -1t *.fits | head -n 2) ../recoveries/

echo "=== [DISK_GAP] Pipeline complete: $(date) ==="
"""

# ─────────────────────────────────────────────────────────────────────────────
for disk, gap_ix, subfolder in GAPS:
    gap_str     = str(gap_ix)
    disk_gap    = f'{disk}_gap{gap_ix}'      # e.g. AA_Tau_gap0
    folder_path = os.path.join(BASE_DIR, subfolder, disk_gap)
    os.makedirs(folder_path, exist_ok=True)

    # ── 1. injectloop.py ─────────────────────────────────────────────────
    t = read_tmpl('AA_Tau_gap0_injectloop.py')
    t = t.replace('import diskdictionary_eqC1 as disk',
                  'import diskdictionary_newgap as disk')
    t = t.replace("target  = 'AA_Tau'    # CSD name",
                  f"target  = '{disk}'    # CSD name")
    t = t.replace('gap_ix  = 0           # which gap',
                  f'gap_ix  = {gap_ix}           # which gap')
    write_lf(os.path.join(folder_path, f'{disk_gap}_injectloop.py'), t)

    # ── 2. imageloop.py ──────────────────────────────────────────────────
    t = read_tmpl('AA_Tau_robust0_5_gap0_imageloop.py')
    t = t.replace('import diskdictionary_eqC1 as disk',
                  'import diskdictionary_newgap as disk')
    t = t.replace('target, gap_ix, subsuf = "AA_Tau", "0", "0"',
                  f'target, gap_ix, subsuf = "{disk}", "{gap_str}", "0"')
    write_lf(os.path.join(folder_path,
                          f'{disk}_robust0_5_gap{gap_ix}_imageloop.py'), t)

    # ── 3. prepimaging.py ────────────────────────────────────────────────
    t = read_tmpl('prepimaging.py')
    t = t.replace('import diskdictionary_eqC1 as disk',
                  'import diskdictionary_newgap as disk')
    t = t.replace("target, gap_ix, subsuf = 'AA_Tau', '0', '0'",
                  f"target, gap_ix, subsuf = '{disk}', '{gap_str}', '0'")
    write_lf(os.path.join(folder_path, 'prepimaging.py'), t)

    # ── 4. prepimaging_casa.py ───────────────────────────────────────────
    t = read_tmpl('prepimaging_casa.py')
    t = t.replace('import diskdictionary_eqC1 as disk',
                  'import diskdictionary_newgap as disk')
    write_lf(os.path.join(folder_path, 'prepimaging_casa.py'), t)

    # ── 5. mask_to_casa.py (no changes) ──────────────────────────────────
    src = read_tmpl('mask_to_casa.py')
    write_lf(os.path.join(folder_path, 'mask_to_casa.py'), src)

    # ── 6. recover_loop.py ───────────────────────────────────────────────
    t = read_tmpl('recover_loop.py')
    t = t.replace('import diskdictionary_eqC1 as disk',
                  'import diskdictionary_newgap as disk')
    t = t.replace("target = 'AA_Tau'", f"target = '{disk}'")
    t = t.replace('gap    = 0',        f'gap    = {gap_ix}')
    t = t.replace('# Load gap properties (wgap = sigma_gap from eqC1 fit)',
                  '# Load gap properties (wgap = outer_edge - inner_edge from manual click)')
    write_lf(os.path.join(folder_path, 'recover_loop.py'), t)

    # ── 7. run_pipeline.sh ───────────────────────────────────────────────
    sh = SH_TEMPLATE
    sh = sh.replace('SUBFOLDER',  subfolder)
    sh = sh.replace('DISK_GAP',   disk_gap)
    sh = sh.replace('DISK_robust0_5_GAPIX',
                    f'{disk}_robust0_5_gap{gap_ix}')
    sh = sh.replace('DISK_',      disk + '_')   # DISK_ prefix replacements done
    write_lf(os.path.join(folder_path, 'run_pipeline.sh'), sh)

    # ── 8. assess_recovery.ipynb ─────────────────────────────────────────
    nb = NB_TEMPLATE
    # sys.path for local import
    nb = nb.replace(r'd:\\CPD_MPIA\\HPC_eqC1_gap',
                    r'd:\\CPD_MPIA\\HPC_huang_gap')
    # dictionary import
    nb = nb.replace('import diskdictionary_eqC1 as disk',
                    'import diskdictionary_newgap as disk')
    # target/gap_ix cell
    nb = nb.replace("target, gap_ix, subsuf = 'AA_Tau', '0', '0'",
                    f"target, gap_ix, subsuf = '{disk}', '{gap_str}', '0'")
    # imdir: eqC1 always uses gap/ in the template path
    nb = nb.replace('D:/CPD_MPIA/HPC_eqC1_gap/gap/',
                    f'D:/CPD_MPIA/HPC_huang_gap/{subfolder}/')
    write_lf(os.path.join(folder_path, 'assess_recovery.ipynb'), nb)

    print(f'  created  {subfolder}/{disk_gap}/')

print('\nAll 17 folders created successfully.')
