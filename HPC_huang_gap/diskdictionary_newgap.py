# diskdictionary_newgap.py
# rgap / wgap from manual visual click in Injection_Recovery_Revised/new_gap.ipynb
#   rgap = gap_dip r_arcsec (first click on the gap minimum)
#   wgap = outer_edge - inner_edge  (arcsec; full width of the injection zone)
#   injection zone = [rgap - 0.5*wgap, rgap + 0.5*wgap]   (covers [inner_edge, outer_edge])
#   recovery search annulus = [rgap - wgap, rgap + wgap]   (2x wider than injection zone)
#
# Click data saved in:  Injection_Recovery_Revised/gap_clicks.csv
# Gap fitting notebook: Injection_Recovery_Revised/new_gap.ipynb
#
# Imaging parameters (crobust, RMS, cthresh, rout, gscales) unchanged
#   from diskdictionary_eqC1.py / diskdictionaryrm0_5.py.
# gthresh = 2 × RMS for all disks (CQ_Tau and J1604 were missing this in eqC1).
# All geometry parameters (PA, incl, dx, dy, cmask, ...) unchanged.

import numpy as np

disk = {
    'AA_Tau': {
        'PA': 93.77079777,
        'R90': np.float64(1.035),
        'RMS': np.float64(42.945290260831825),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[04:34:55.420, +24:28:53.034], [2arcsec, '
                 '1.0439459414332437arcsec], 93.77079777deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.215mJy',
        'distance': 135,
        'dx': -0.00545897,
        'dy': 0.00482739,
        'gscales': [0, 5],
        'gthresh': '0.086mJy', 
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 58.53531224,
        'label': 'AA Tau',
        'lstar': 1.1,
        'mstar': 0.79,
        'name': 'AA_Tau',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.13", inner=0.05", outer=0.23"  → outer ring gap
        #   gap1: gap_dip=0.57", inner=0.51", outer=0.61"  → outer ring gap
        'rgap': [0.13, 0.57],
        'wgap': [0.18, 0.10],
        'rout': np.float64(0.8539673108605426),
    },

    'CQ_Tau': {
        'PA': 53.87180444,
        'R90': np.float64(0.4894),
        'RMS': np.float64(39.13380714948289),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[05:35:58.467, +24:44:54.091], [2arcsec, '
                 '1.633432101799876arcsec], 53.87180444deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.196mJy',
        'distance': 149,
        'dx': -0.00871044,
        'dy': 0.0009941,
        'gscales': [0, 5],
        'gthresh': '0.078mJy',    # 2 × RMS = 2 × 39.134 µJy
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 35.2426038,
        'label': 'CQ Tau',
        'lstar': 10,
        'mstar': 1.4,
        'name': 'CQ_Tau',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.05", inner=0.01", outer=0.09"  → inner cavity
        'rgap': [0.05],
        'wgap': [0.08],
        'rout': np.float64(0.6461462387813894),
    },

    'DM_Tau': {
        'PA': 155.59975598,
        'R90': np.float64(1.402),
        'RMS': np.float64(36.36885594460182),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[04:33:48.733, +18:10:09.973], [2arcsec, '
                 '1.6187017651640194arcsec], 155.59975598deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.182mJy',
        'distance': 144,
        'dx': -0.00551499,
        'dy': -0.00658999,
        'gscales': [0, 5],
        'gthresh': '0.073mJy',
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 35.96744071,
        'label': 'DM Tau',
        'lstar': 0.24,
        'mstar': 0.45,
        'name': 'DM_Tau',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.07", inner=0.05", outer=0.11"  → inner cavity
        #   gap1: gap_dip=0.49", inner=0.47", outer=0.57"  → outer ring gap (~70 AU)
        'rgap': [0.07, 0.49],
        'wgap': [0.06, 0.10],
        'rout': np.float64(0.8455571634694635),
    },

    'HD_135344B': {
        'PA': 28.92070668,
        'R90': np.float64(0.6681),
        'RMS': np.float64(44.64330959308427),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[15:15:48.446, -37:09:16.024], [2arcsec, '
                 '1.8704829781176846arcsec], 28.92070668deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.223mJy',
        'distance': 135,
        'dx': 0.0007974,
        'dy': -0.00320815,
        'gscales': [0, 5],
        'gthresh': '0.089mJy',
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 20.73280574,
        'label': 'HD 135344B',
        'lstar': 6.7,
        'mstar': 1.61,
        'name': 'HD_135344B',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.51", inner=0.47", outer=0.55"  → outer ring gap
        #   gap1: gap_dip=0.11", inner=0.03", outer=0.21"  → inner cavity
        'rgap': [0.51, 0.11],
        'wgap': [0.08, 0.18],
        'rout': np.float64(0.0112374890595675),
    },

    'J1604': {
        'PA': 123.24221853,
        'R90': np.float64(0.7776),
        'RMS': np.float64(39.42582043237053),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[16:04:21.642, -21:30:29.058], [2arcsec, '
                 '1.9768678096037415arcsec], 123.24221853deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.197mJy',
        'distance': 145,
        'dx': -0.07482211,
        'dy': -0.01666589,
        'gscales': [0, 5],
        'gthresh': '0.079mJy',    # 2 × RMS = 2 × 39.426 µJy
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 8.7226911,
        'label': 'J1604',
        'lstar': 0.76,
        'mstar': 1.29,
        'name': 'J1604',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.37", inner=0.33", outer=0.41"  → inner cavity
        'rgap': [0.37],
        'wgap': [0.08],
        'rout': np.float64(0.0118885850533845),
    },

    'J1615': {
        'PA': 146.14327945,
        'R90': np.float64(1.09),
        'RMS': np.float64(35.858654882758856),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[16:15:20.234, -32:55:05.099], [2arcsec, '
                 '1.3614990911539273arcsec], 146.14327945deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.179mJy',
        'distance': 156,
        'dx': -0.04431726,
        'dy': -0.00588429,
        'gscales': [0, 5],
        'gthresh': '0.072mJy',
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 47.09775702,
        'label': 'J1615',
        'lstar': 1.07,
        'mstar': 1.14,
        'name': 'J1615',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.55", inner=0.53", outer=0.61"  → outer ring gap
        #   gap1: gap_dip=0.05", inner=0.01", outer=0.09"  → inner cavity
        'rgap': [0.55, 0.05],
        'wgap': [0.08, 0.08],
        'rout': np.float64(1.2796303443610049),
    },

    'J1852': {
        'PA': 117.61494041,
        'R90': np.float64(0.475),
        'RMS': np.float64(34.631972084753215),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[18:52:17.301, -37:00:11.949], [2arcsec, '
                 '1.6868119485868183arcsec], 117.61494041deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.173mJy',
        'distance': 147,
        'dx': -0.02340882,
        'dy': 0.00190753,
        'gscales': [0, 5],
        'gthresh': '0.069mJy',
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 32.4984507,
        'label': 'J1852',
        'lstar': 0.6,
        'mstar': 1.03,
        'name': 'J1852',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.19", inner=0.15", outer=0.23"  → inner cavity
        'rgap': [0.19],
        'wgap': [0.08],
        'rout': np.float64(0.0123747196048515),
    },

    'LkCa_15': {
        'PA': 61.57232522,
        'R90': np.float64(0.9943),
        'RMS': np.float64(34.40072396188043),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[04:39:17.791, +22:21:03.390], [2arcsec, '
                 '1.2695975167543332arcsec], 61.57232522deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.172mJy',
        'distance': 156,
        'dx': -0.01684397,
        'dy': 0.02082818,
        'gscales': [0, 5],
        'gthresh': '0.069mJy',
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 50.59493965,
        'label': 'LkCa 15',
        'lstar': 1,
        'mstar': 1.14,
        'name': 'LkCa_15',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.57", inner=0.55", outer=0.59"  → outer ring gap
        #   gap1: gap_dip=0.07", inner=0.03", outer=0.11"  → inner cavity
        'rgap': [0.57, 0.07],
        'wgap': [0.04, 0.08],
        'rout': np.float64(0.0093515329062945),
    },

    'MWC_758': {
        'PA': 76.17365568,
        'R90': np.float64(0.5862),
        'RMS': np.float64(57.1458695048932),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[05:30:27.529, +25:19:57.076], [2arcsec, '
                 '1.9839420811003194arcsec], 76.17365568deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.286mJy',
        'distance': 156,
        'dx': 0.02548176,
        'dy': 0.01841853,
        'gscales': [0, 5],
        'gthresh': '0.114mJy',
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 7.26537891,
        'label': 'MWC 758',
        'lstar': 10.4,
        'mstar': 1.4,
        'name': 'MWC_758',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.13", inner=0.09", outer=0.17"  → inner cavity
        #   gap1: gap_dip=0.41", inner=0.39", outer=0.47"  → outer ring gap
        'rgap': [0.13, 0.41],
        'wgap': [0.08, 0.08],
        'rout': np.float64(0.012682148255406),
    },

    'SY_Cha': {
        'PA': 165.76511321,
        'R90': np.float64(1.094),
        'RMS': np.float64(54.445092246169224),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[10:56:30.388, -77:11:39.402], [2arcsec, '
                 '1.2410252268065851arcsec], 165.76511321deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.272mJy',
        'distance': 182,
        'dx': -0.01265537,
        'dy': 0.02815886,
        'gscales': [0, 5],
        'gthresh': '0.109mJy',
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 51.64642211,
        'label': 'SY Cha',
        'lstar': 0.55,
        'mstar': 0.77,
        'name': 'SY_Cha',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.19", inner=0.11", outer=0.27"  → ring gap
        'rgap': [0.19],
        'wgap': [0.16],
        'rout': np.float64(0.07987987529486999),
    },

    'V4046_Sgr': {
        'PA': 76.02083693,
        'R90': np.float64(0.852),
        'RMS': np.float64(36.36974361143075),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[18:14:10.482, -32:47:34.517], [2arcsec, '
                 '1.6704810703258652arcsec], 76.02083693deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.182mJy',
        'distance': 72,
        'dx': -0.05093653,
        'dy': -0.04518314,
        'gscales': [0, 5],
        'gthresh': '0.073mJy',
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 33.35910733,
        'label': 'V4046 Sgr',
        'lstar': 0.5,
        'mstar': 1.73,
        'name': 'V4046_Sgr',
        # manual click (new_gap.ipynb):
        #   gap0: gap_dip=0.09", inner=0.05", outer=0.13"  → inner cavity
        #   gap1: gap_dip=0.27", inner=0.19", outer=0.35"  → outer ring gap
        'rgap': [0.09, 0.27],
        'wgap': [0.08, 0.16],
        'rout': np.float64(1.0326922489331056),
    },
}
