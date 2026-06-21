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
        'dgap': [6.50507, 200.0],
        'rgap': [0.13565, 0.57],
        'wgap': [0.04485, 0.12141],
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
        'gthresh': [],
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
        #   gap0: rgap=0.08", sigma=0.02847", delta=3.677  [inner cavity]
        'dgap': [3.67652],
        'rgap': [0.08],
        'wgap': [0.02847],
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
        'dgap': [1.62421, 25.13334],
        'rgap': [0.49531, 0.09],
        'wgap': [0.039, 0.07899],
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
        'gthresh': [],
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
        'dgap': [1.41114, 1.31547],
        'rgap': [0.54, 0.04],
        'wgap': [0.05492, 0.05178],
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
        'dgap': [1.21326, 11.57376],
        'rgap': [0.56, 0.06],
        'wgap': [0.03638, 0.06036],
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
        'dgap': [15.96036],
        'rgap': [0.2],
        'wgap': [0.06493],
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
