# diskdictionary_eqC1.py
# rgap / wgap (= sigma_gap) from Eq. C1 brightness temperature profile fits
#   fitted to the CLEAN radial profile (robust = -0.5, units mJy/beam)
#   fit model: I(r) = I0*(r/0.1")^(-q) / (1 + Gamma)
#              Gamma = (delta-1)*exp[-0.5*((r-r_gap)/sigma_gap)^2]
#   wgap stored here is sigma_gap (Gaussian half-width of the depletion)
#   injection zone = r_gap +/- 1*sigma_gap  (ZONE_NSIGMA = 1)
#
# Imaging parameters (crobust, RMS, cthresh, gthresh, rout) from
#   diskdictionaryrm0_5.py  (robust = -0.5)
# All other geometry parameters (PA, incl, cmask, dx, dy, ...) unchanged.

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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.13565", sigma=0.04485", delta=6.505
        #   gap1: rgap=0.57",    sigma=0.12141", delta=200.0  [WARNING: hit upper bound]
        # rkink/wkink: kink-candidate position, kept separate from the real eqC1 gaps.
        #   rkink = Pinte+2025 Table 2 r_planet (au) / distance (arcsec), same as CANDIDATE_R in the
        #   kink folders (azimuth: CANDIDATE_AZ, from the Figure 5 blue dot, verify_figure5_blue_dot.ipynb).
        #   wkink is only a mask-sizing radius (0.5x beam) for custom_mask(..., feature='kink').
        'dgap': [6.50507, 200.0],
        'rgap': [0.13565, 0.57],
        'wgap': [0.04485, 0.12141],
        'rkink': [0.5926],
        'wkink': [0.0316],
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
        'cthresh': '0.196mJy', # RMS is 39.13 mJy/beam, so 5-sigma is 0.196 mJy
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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.04", sigma=0.05503", delta=15.228
        'dgap': [15.2282],
        'rgap': [0.04],
        'wgap': [0.05503],
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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.08",  sigma=0.02847", delta=3.677   [inner cavity]
        #   gap1: rgap=0.48",  sigma=0.05021", delta=1.484   [outer ring gap ~70 AU]
        'dgap': [3.67652, 1.48446],
        'rgap': [0.08, 0.48],
        'wgap': [0.02847, 0.05021],
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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.49531", sigma=0.039",   delta=1.624  [outer ring gap]
        #   gap1: rgap=0.09",    sigma=0.07899",  delta=25.133 [inner cavity]
        # indices 2-4: NOT eqC1 gap fits -- fixed point-candidate positions from
        # Table 1 (CO-tracer planet candidates C1, C2, DF), used only so the
        # existing custom_mask/injectloop/recover_loop machinery (which is
        # written around a rgap/wgap zone) can size a CASA clean mask and
        # injection point per candidate. wgap here is just a mask-sizing
        # radius, not a fitted gap width -- see HD_135344B_C1/C2/DF folders,
        # where injectloop.py overrides r_cpd/az_cpd to the exact candidate
        # position rather than drawing from this zone.
        #   C1: R=0.30", phi=36deg    (41 au,  36deg)
        #   C2: R=0.54", phi=15deg    (73 au,  15deg)
        #   DF: R=0.69", phi=-133deg  (95 au, -133deg)
        'dgap': [1.62421, 25.13334, 1.0, 1.0, 1.0],
        'rgap': [0.49531, 0.09, 0.30, 0.54, 0.69],
        'wgap': [0.039, 0.07899, 0.03, 0.03, 0.03],
        'rout': np.float64(0.0112374890595675),
    },

    'HD_143006': {
        'PA': 7.53236754,
        'R90': np.float64(0.4778),
        'RMS': np.float64(45.40990266832523),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[15:58:36.913, -22:57:15.221], [2arcsec, '
                 '1.8944886188220467arcsec], 7.53236754deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.227mJy',
        'distance': 167,
        'dx': 0.00826703,
        'dy': 0.02642291,
        'gscales': [0, 5],
        'gthresh': '0.091mJy',
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 18.69391735,
        'label': 'HD 143006',
        'lstar': 3.8,
        'mstar': 1.56,
        'name': 'HD_143006',
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.13935", sigma=0.03495", delta=3.167  [gap between inner disk and 0.25" ring]
        #   gap1: rgap=0.31635", sigma=0.02283", delta=1.111  [shallow; fitted with q<0 since the
        #         0.39" ring is brighter than the 0.25" ring]
        'dgap': [3.16653, 1.11104],
        'rgap': [0.13935, 0.31635],
        'wgap': [0.03495, 0.02283],
        'rout': np.float64(0.573863300494785),
    },

    'HD_34282': {
        'PA': 117.79519232,
        'R90': np.float64(0.7759),
        'RMS': np.float64(39.72522608819418),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[05:16:00.477, -09:48:35.394], [2arcsec, '
                 '1.005527473149872arcsec], 117.79519232deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.199mJy',
        'distance': 309,
        'dx': 0.01282042,
        'dy': 0.00837812,
        'gscales': [0, 5],
        'gthresh': '0.079mJy',
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 59.81698359,
        'label': 'HD 34282',
        'lstar': 108,
        'mstar': 1.61,
        'name': 'HD_34282',
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.05", sigma=0.04354", delta=4.525  [inner cavity, beam_0.5fwhm convention]
        'dgap': [4.52491],
        'rgap': [0.05],
        'wgap': [0.04354],
        'rout': np.float64(0.008326621726155),
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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.36", sigma=0.05515", delta=46.030 [inner cavity]
        'dgap': [46.0303],
        'rgap': [0.36],
        'wgap': [0.05515],
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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.54", sigma=0.05492", delta=1.411  [outer ring gap]
        #   gap1: rgap=0.04", sigma=0.05178", delta=1.315  [inner cavity]
        # rkink/wkink: kink-candidate position, kept separate from the real eqC1 gaps.
        #   rkink = Pinte+2025 Table 2 r_planet (au) / distance (arcsec), same as CANDIDATE_R in the
        #   kink folders (azimuth: CANDIDATE_AZ, from the Figure 5 blue dot, verify_figure5_blue_dot.ipynb).
        #   wkink is only a mask-sizing radius (0.5x beam) for custom_mask(..., feature='kink').
        'dgap': [1.41114, 1.31547],
        'rgap': [0.54, 0.04],
        'wgap': [0.05492, 0.05178],
        'rkink': [1.9872],
        'wkink': [0.0450],
        'rout': np.float64(1.2796303443610049),
    },

    'J1842': {
        'PA': 26.35016474,
        'R90': np.float64(0.5645),
        'RMS': np.float64(43.370182),
        'ccycleniter': 300,
        'cgain': 0.3,
        'cmask': 'ellipse[[18:42:57.981, -35:32:42.827], [2arcsec, '
                 '1.549481012108344arcsec], 26.35016474deg]',
        'crobust': -0.5,
        'cscale': [0, 8, 15, 30, 80],
        'ctaper': [],
        'cthresh': '0.217mJy',   # 5 x RMS
        'distance': 151,
        'dx': -0.00316491,
        'dy': -0.03068915,
        'gscales': [0, 5],
        'gthresh': '0.087mJy',   # 2 x RMS
        'hyp-Ncoll': 300,
        'hyp-alpha': 1.3,
        'hyp-wsmth': 0.1,
        'incl': 39.21848812,
        'label': 'J1842',
        'lstar': 0.8,
        'mstar': 1.07,
        'name': 'J1842',
        # No eqC1 gap fit exists for J1842 yet -- this disk was never part of
        # the gap/inner_cavity injection-recovery batches. Added here only to
        # support the kink-candidate injection-recovery test (rkink/wkink below).
        # PA/incl/dx/dy: galario fit (J1842_geometrical_parameters_continuum_galario.txt)
        # RMS: measured from images_data_different_robust robust=-0.5 image (r>3" annulus)
        # R90: from J1842_continuum_radii.txt; rout: no frank fit available, R90 used as fallback
        # rkink/wkink: kink-candidate position, kept separate from the real eqC1 gaps.
        #   rkink = Pinte+2025 Table 2 r_planet (au) / distance (arcsec), same as CANDIDATE_R in the
        #   kink folders (azimuth: CANDIDATE_AZ, from the Figure 5 blue dot, verify_figure5_blue_dot.ipynb).
        #   wkink is only a mask-sizing radius (0.5x beam) for custom_mask(..., feature='kink').
        'dgap': [],
        'rgap': [],
        'wgap': [],
        'rkink': [0.6954],
        'wkink': [0.0416],
        'rout': np.float64(0.5645),
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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.17", sigma=0.04294", delta=34.615 [inner cavity]
        'dgap': [34.61492],
        'rgap': [0.17],
        'wgap': [0.04294],
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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.56", sigma=0.03638", delta=1.213  [outer ring gap]
        #   gap1: rgap=0.06", sigma=0.06036", delta=11.574 [inner cavity]
        # rkink/wkink: kink-candidate position, kept separate from the real eqC1 gaps.
        #   rkink = Pinte+2025 Table 2 r_planet (au) / distance (arcsec), same as CANDIDATE_R in the
        #   kink folders (azimuth: CANDIDATE_AZ, from the Figure 5 blue dot, verify_figure5_blue_dot.ipynb).
        #   wkink is only a mask-sizing radius (0.5x beam) for custom_mask(..., feature='kink').
        'dgap': [1.21326, 11.57376],
        'rgap': [0.56, 0.06],
        'wgap': [0.03638, 0.06036],
        'rkink': [1.5385],
        'wkink': [0.0332],
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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.12",   sigma=0.04897", delta=32.360 [inner cavity]
        #   gap1: rgap=0.4034", sigma=0.04088", delta=1.298  [outer ring gap]
        'dgap': [32.3602, 1.29829],
        'rgap': [0.12, 0.4034],
        'wgap': [0.04897, 0.04088],
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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.2", sigma=0.06493", delta=15.960 [inner cavity]
        # rkink/wkink: kink-candidate position, kept separate from the real eqC1 gaps.
        #   rkink = Pinte+2025 Table 2 r_planet (au) / distance (arcsec), same as CANDIDATE_R in the
        #   kink folders (azimuth: CANDIDATE_AZ, from the Figure 5 blue dot, verify_figure5_blue_dot.ipynb).
        #   wkink is only a mask-sizing radius (0.5x beam) for custom_mask(..., feature='kink').
        'dgap': [15.96036],
        'rgap': [0.2],
        'wgap': [0.06493],
        'rkink': [0.7692],
        'wkink': [0.0396],
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
        # eqC1 fit (robust=-0.5 CLEAN profile):
        #   gap0: rgap=0.08",    sigma=0.03785", delta=4.232  [inner cavity]
        #   gap1: rgap=0.27754", sigma=0.03684", delta=6.208  [outer ring gap]
        'dgap': [4.23241, 6.20796],
        'rgap': [0.08, 0.27754],
        'wgap': [0.03785, 0.03684],
        'rout': np.float64(1.0326922489331056),
    },
}
