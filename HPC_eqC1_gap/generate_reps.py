"""
generate_reps.py
Creates 9 independent repeat copies (_rep1 .. _rep9) of each disk/gap pipeline
folder under HPC_eqC1_gap/gap/ and HPC_eqC1_gap/inner_cavity/, mirroring the
DSHARP_rep/HD143006_gap1_rep{1..9} pattern.

Each rep is a full self-contained copy of the base folder's pipeline scripts
(own WORKDIR, so it builds its own mask/psf and runs entirely independently).
Every rep runs subsuf='0' internally, same as the base run; combined with the
base folder that's 10 runs x 50 mocks/flux-bin = 500 mocks/flux-bin.

Because all disks share the same measurement_set_spavg/ tree on the HPC side
(see prepimaging_casa.py / imageloop.py), the intermediate MS built via
ImportMS() is named after resid_suffix -- without a per-rep tag, N reps of the
same disk/gap running in parallel would race on that same shared filename.
So this script inserts a ".repN" tag into resid_suffix (prepimaging_casa.py,
the imageloop) and into im_file (recover_loop.py) so it matches the images
the imageloop actually produces.

run_pipeline.sh also gets a per-rep isolated $HOME (like DSHARP_rep's reps),
since the shared $HOME otherwise causes CASA ipython-history "database is
locked" crashes when many CASA sessions launch at once.

Run once locally; then rsync the whole HPC_eqC1_gap/ tree to the cluster.
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HPC_BASE = "/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap"
N_REPS = 9

# (disk_name, gap_ix, subfolder) -- must match generate_scripts.py TARGETS
TARGETS = [
    # --- ring gaps ---
    ("HD_135344B", 0, "gap"),
    ("J1615",      0, "gap"),
    ("LkCa_15",    0, "gap"),
    ("MWC_758",    1, "gap"),
    ("V4046_Sgr",  1, "gap"),
    ("AA_Tau",     0, "gap"),
    ("AA_Tau",     1, "gap"),
    # --- inner cavities ---
    ("DM_Tau",       0, "inner_cavity"),
    ("HD_135344B",   1, "inner_cavity"),
    ("J1604",        0, "inner_cavity"),
    ("J1615",        1, "inner_cavity"),
    ("J1852",        0, "inner_cavity"),
    ("LkCa_15",      1, "inner_cavity"),
    ("MWC_758",      0, "inner_cavity"),
    ("SY_Cha",       0, "inner_cavity"),
    ("V4046_Sgr",    0, "inner_cavity"),
    ("CQ_Tau",       0, "inner_cavity"),
]

# One-off targets outside the standard 17 (e.g. the DM_Tau/SY_Cha "3 ways"
# comparison folders documented in dmtau_sycha_3ways.md) that also want reps.
EXTRA_TARGETS = [
    ("DM_Tau", 1, "gap"),  # gap/DM_Tau_gap1 -- comparison run, not in TARGETS
    ("SY_Cha", 0, "gap"),  # gap/SY_Cha_gap0 -- comparison run, not in TARGETS
]

HOME_ISOLATION_BLOCK = """
# isolate this rep's CASA ipython history db (shared $HOME causes
# "database is locked" crashes when multiple CASA sessions launch at once);
# symlink in the real measures data so CASA does not try to re-download it
export HOME="${WORKDIR}/.casa_home"
mkdir -p "$HOME/.casa"
ln -sf "/home/vawelke/.casa/data" "$HOME/.casa/data" 2>/dev/null
ln -sf "/home/vawelke/.casa/measures.d" "$HOME/.casa/measures.d" 2>/dev/null
"""


def tag_resid_suffix(content, rep):
    """Insert .repN right after the 'gap'+gap_ix+'. piece of resid_suffix."""
    new_content, n = re.subn(
        r"(resid_suffix = 'gap'\+gap_ix\+'\.)",
        rf"resid_suffix = 'gap'+gap_ix+'.rep{rep}.",
        content,
    )
    if n != 1:
        raise RuntimeError("expected exactly one resid_suffix line to tag")
    return new_content


def tag_im_file(content, rep):
    pattern = r"(im_file = target \+ '_gap' \+ str\(gap\) \+ '\.)"
    new_content, n = re.subn(
        pattern,
        rf"im_file = target + '_gap' + str(gap) + '.rep{rep}.",
        content,
    )
    if n != 1:
        raise RuntimeError("expected exactly one im_file line to tag")
    return new_content


def make_run_pipeline_sh(content, base_workdir, rep_workdir):
    new_content, n = re.subn(
        re.escape(f'WORKDIR="{base_workdir}"'),
        f'WORKDIR="{rep_workdir}"',
        content,
    )
    if n != 1:
        raise RuntimeError("expected exactly one WORKDIR line to replace")

    marker = 'export CASA_CONFIG="/nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py"\n'
    if marker not in new_content:
        raise RuntimeError("could not find CASA_CONFIG export line")
    new_content = new_content.replace(marker, marker + HOME_ISOLATION_BLOCK, 1)
    return new_content


def main():
    created = []
    for target, gap_ix, subfolder in TARGETS + EXTRA_TARGETS:
        folder_name = f"{target}_gap{gap_ix}"
        src_dir = os.path.join(BASE_DIR, subfolder, folder_name)
        if not os.path.isdir(src_dir):
            raise RuntimeError(f"base folder missing: {src_dir}")

        injectloop_fname = f"{folder_name}_injectloop.py"
        imageloop_fname = f"{target}_robust0_5_gap{gap_ix}_imageloop.py"
        base_workdir = f"{HPC_BASE}/{subfolder}/{folder_name}"

        for rep in range(1, N_REPS + 1):
            rep_name = f"{folder_name}_rep{rep}"
            dst_dir = os.path.join(BASE_DIR, subfolder, rep_name)
            os.makedirs(dst_dir, exist_ok=True)
            rep_workdir = f"{HPC_BASE}/{subfolder}/{rep_name}"

            for fname in [injectloop_fname, "prepimaging.py", "prepimaging_casa.py",
                          imageloop_fname, "recover_loop.py", "run_pipeline.sh",
                          "mask_to_casa.py"]:
                src_path = os.path.join(src_dir, fname)
                with open(src_path, "r", encoding="utf-8", newline="") as fh:
                    content = fh.read()

                if fname == "prepimaging_casa.py" or fname == imageloop_fname:
                    content = tag_resid_suffix(content, rep)
                elif fname == "recover_loop.py":
                    content = tag_im_file(content, rep)
                elif fname == "run_pipeline.sh":
                    content = make_run_pipeline_sh(content, base_workdir, rep_workdir)

                dst_path = os.path.join(dst_dir, fname)
                with open(dst_path, "w", encoding="utf-8", newline="") as fh:
                    fh.write(content)
                created.append(os.path.relpath(dst_path, BASE_DIR))

    n_targets = len(TARGETS) + len(EXTRA_TARGETS)
    print(f"Created {len(created)} files across "
          f"{n_targets * N_REPS} rep folders ({n_targets} disks x {N_REPS} reps).")


if __name__ == "__main__":
    main()
