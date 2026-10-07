import numpy as np

# HD_135344B geometry
PA = 28.92070668
incl = 20.73280574
PAr, inclr = np.radians(PA), np.radians(incl)


def az_to_sky(r, az_deg):
    """This pipeline's disk-plane-azimuth -> sky (RA_offset, Dec_offset)
    convention, verified against recover_loop.py's inverse transform
    elsewhere in this project (matches where injected sources are actually
    recovered)."""
    az = np.radians(az_deg)
    x = r * np.sin(az) * np.sin(PAr) + r * np.cos(az) * np.cos(inclr) * np.cos(PAr)
    y = r * np.sin(az) * np.cos(PAr) - r * np.cos(az) * np.cos(inclr) * np.sin(PAr)
    return x, y


def bearing(x, y):
    """Compass bearing (0=N, 90=E, 180=S, 270=W) for a sky offset
    (x=RA offset East-positive, y=Dec offset North-positive)."""
    return np.degrees(np.arctan2(x, y)) % 360


candidates = {'C1': (0.30, 36.0), 'C2': (0.54, 15.0), 'DF': (0.69, -133.0)}

print(f'PA={PA:.2f}  incl={incl:.2f}  (nearly face-on)')
print()

# --- relative-separation check: convention-independent, uses raw phi only ---
print('--- relative azimuthal separation (convention-independent) ---')
labels = list(candidates)
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        a, b = labels[i], labels[j]
        daz = abs(candidates[a][1] - candidates[b][1])
        daz = min(daz, 360 - daz)
        print(f'  {a} - {b}: {daz:.1f} deg apart')
print()

# --- this pipeline's absolute sky-bearing interpretation ---
print('--- this pipeline\'s sky-bearing interpretation ---')
print(f'{"label":>5} {"r":>6} {"az(phi)":>8} {"RA_offset(E+)":>14} {"Dec_offset(N+)":>15} {"bearing":>9}')
for label, (r, az) in candidates.items():
    x, y = az_to_sky(r, az)
    b = bearing(x, y)
    print(f'{label:>5} {r:6.2f} {az:8.1f} {x:14.4f} {y:15.4f} {b:9.1f}')
print()

print('--- reference bearings for az=0/90/180/-90 (this pipeline) ---')
for az in [0, 90, 180, -90]:
    x, y = az_to_sky(1.0, az)
    b = bearing(x, y)
    print(f'  az={az:5.0f} deg  ->  RA_offset={x:7.3f}  Dec_offset={y:7.3f}  bearing={b:6.1f} deg')

print()
print('To fully confirm: find the paper\'s stated phi reference direction and')
print('rotation sense (e.g. "measured counterclockwise from North") and compare')
print('against the az=90 row above -- our az=90 sits at bearing %.1f deg.' %
      bearing(*az_to_sky(1.0, 90)))
