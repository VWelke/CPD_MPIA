import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# ── directories to compare ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

runs = {
    'inj_rev/J1852_gap0\n(rout, diskdict_pix)':
        os.path.join(BASE, 'J1852_gap0', 'mprofiles'),
    'J1852_gap0_r0_5\n(rout, diskdict_pix, rob0.5)':
        os.path.join(BASE, 'J1852_gap0_r0_5', 'mprofiles'),
    'inj_rev/r0_5/J1852_gap0\n(R90, diskdict_r90)':
        os.path.join(BASE, 'inj_rev', 'r0_5', 'J1852_gap0', 'mprofiles'),
}

# ── disk parameters (from diskdictionaryr0_5) ───────────────────────────────
rout  = 0.7218   # arcsec
R90   = 0.475    # arcsec
rgap  = 0.120    # arcsec  (gap centre)
wgap  = 0.195    # arcsec  (gap half-width)

N_SAMPLE = 6     # profiles to overlay per run

# ── plot ────────────────────────────────────────────────────────────────────
colors = ['tab:blue', 'tab:orange', 'tab:green']
fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5), sharey=True)

for ax, (label, mdir), color in zip(axes, runs.items(), colors):
    files = sorted(glob.glob(os.path.join(mdir, '*_frank_profile_fit.txt')))
    if not files:
        ax.text(0.5, 0.5, 'no mprofiles\nfound', ha='center', va='center',
                transform=ax.transAxes, color='red')
        ax.set_title(label)
        continue

    sample = files[:N_SAMPLE]
    for i, fpath in enumerate(sample):
        try:
            data = np.loadtxt(fpath)
            r, I = data[:, 0], data[:, 1] * 1e-10  # Jy/sr → scaled for display
            ax.plot(r, I, color=color, alpha=0.5, lw=1,
                    label=os.path.basename(fpath)[:20] if i == 0 else None)
        except Exception as e:
            print(f'  skip {fpath}: {e}')

    # mark key radii
    ax.axvline(rout,          color='red',    ls='--', lw=1.2, label=f'rout={rout}"')
    ax.axvline(R90,           color='purple', ls='--', lw=1.2, label=f'R90={R90}"')
    ax.axvline(rgap - wgap,   color='grey',   ls=':',  lw=1,   label='gap inner/outer')
    ax.axvline(rgap + wgap,   color='grey',   ls=':',  lw=1)
    ax.axvline(2 * rout,      color='red',    ls=':',  lw=1,   label=f'Rmax(rout)={2*rout:.2f}"')
    ax.axvline(2 * R90,       color='purple', ls=':',  lw=1,   label=f'Rmax(R90)={2*R90:.2f}"')

    ax.set_title(label, fontsize=9)
    ax.set_xlabel('r [arcsec]')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, max(2 * rout, 2 * R90) * 1.1)

axes[0].set_ylabel('Brightness [scaled]')
plt.suptitle('J1852 gap0 — frank profile comparison (R90 vs rout Rmax)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'frank_profile_comparison_J1852.png'), dpi=150)
plt.show()
print('saved: frank_profile_comparison_J1852.png')
