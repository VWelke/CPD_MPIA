# This.py is based on the Zhu_modular..ipynb notebook and is used in the Plot.ipynb notebook

# import everything first:
import numpy as np
import astropy.constants as const
import astropy.units as u
from matplotlib import pyplot as plt
import sys
import pickle
from joblib import load
from joblib import Parallel, delayed
import os
import pandas as pd


# ============================
# Physical constants (CGS)
# ============================

G          = 6.674e-8            # gravitational constant [cm^3 g^-1 s^-2]
sigmaB     = 5.670e-5            # Stefan–Boltzmann constant [erg cm^-2 s^-1 K^-4]
kB         = 1.381e-16           # Boltzmann constant [erg K^-1]

# Luminosities
L_sun      = 3.828e33            # solar luminosity [erg/s]
L_jup      = 3.846e30            # Jupiter luminosity [erg/s] (your input)

# Distances
AU         = 1.496e13            # astronomical unit [cm]

# Masses
M_sun      = 1.989e33            # [g]
M_jup      = 1.898e30            # [g]

# Radii
R_jup      = 7.149e9             # Jupiter radius [cm]

# Time
sec_per_yr = 3.156e7             # seconds in one year

# Gas constant
Rgas       = 8.314e7             # erg mol^-1 K^-1  (used in Zhu eq. 6 and 7)



# ============================ 
# Load pre-calculated relations
# ============================

Lp_from_Mp = load('d:/CPD_MPIA/utils/Lp_from_Mp.pkl')
Rp_from_Mp = load('d:/CPD_MPIA/utils/Rp_from_Mp.pkl')


# ============================
# Planck function
# ============================

def B_lambda(lam, T):
    """
    Planck function in wavelength (erg s^-1 cm^-2 Hz^-1 sr^-1)
    lam: wavelength in cm
    T: temperature in K
    """
    h = 6.626e-27  # erg*s
    c = 2.998e10   # cm/s
    k = 1.381e-16  # erg/K
    return (2*h*c**2 / lam**5) / (np.exp(h*c/(lam*k*T)) - 1)


# ============================
# Zhu CPD emission model
# ============================

import numpy as np

# -------------------------------
# PHYSICAL CONSTANTS
# -------------------------------

# assume you already have G, sigmaB, kB, AU, R_jup, M_jup, M_sun, sec_per_yr etc.


# ================================================================
# A. UNITS AND RADIAL GRID
# ================================================================

def convert_inputs(M_star, Mp, Mdot, rp, Lstar):
    M_star = M_star * M_sun
    Mp     = Mp     * M_jup
    Mdot   = Mdot   * M_jup / sec_per_yr # from M_jup/yr to g/s
    rp     = rp     * AU
    Lstar  = Lstar  * L_sun
    Rp = Rp_from_Mp(Mp / M_jup) * R_jup     # [cm]
    # Luminosities
    Lplanet = Lp_from_Mp(Mp / M_jup) * L_sun  # [erg/s]
    return M_star, Mp, Mdot, rp, Lstar, Rp, Lplanet

def radial_grid(Rp, Rout, Nr=200):
    R = np.geomspace(Rp*1.01, Rout, Nr)
    return R


# ================================================================
# B. HEATING: IRRADIATION AND EFFECTIVE TEMPERATURE
# ================================================================

def irradiation_temperature(R, Rp, Mp, Mdot, Lplanet, Lstar, rp, T_ISM):
    # boundary layer accretion luminosity

    L_irr = G*Mp*Mdot/(2*Rp)
    T_irr = ( L_irr / (40*np.pi*sigmaB*R**2) )**0.25

    # intrinsic planet irradiation
    T_irr_p = (0.1 * Lplanet / (4*np.pi*sigmaB*R**2))**0.25

    # stellar irradiation at distance rp
    phi = 0.02
    T_irr_star = ( phi * Lstar / (8*np.pi*sigmaB*rp**2) )**0.25

    T_ext = (T_irr**4 + T_irr_p**4 + T_irr_star**4 + T_ISM**4)**0.25
    return T_ext, T_irr, T_irr_p, T_irr_star


def effective_temperature(R, Rp, Mp, Mdot):
    Teff = ( (3*G*Mp*Mdot)/(8*np.pi*sigmaB*R**3) * (1 - np.sqrt(Rp/R)) )**0.25
    return Teff


# ================================================================
# C. SURFACE DENSITY
# ================================================================

def sigma_viscous(R, Rp, Mp, Mdot, alpha, kappaR, mu):
    # eq. (6) Zhu 2018
    Rgas = 8.314e7
    term = (sigmaB*G*Mp*Mdot**3/(alpha**4*np.pi**3*kappaR))**0.2
    mu_t = (mu/Rgas)**0.8
    fac = (1 - np.sqrt(Rp/R))**(3/5)
    Sigma6 = (2**(7/5)/3**(6/5)) * term * mu_t * fac * R**(-3/5)
    return Sigma6


def sigma_irradiated(R, Mp, Mdot, alpha, mu, T_ext):
    # eq. (7) Zhu 2018
    Rgas = 8.314e7
    Omega = np.sqrt(G*Mp/R**3)
    Sigma7 = (Mdot * mu * Omega) / (3*np.pi*alpha*Rgas*T_ext)
    return Sigma7


def sigma_combined(Sigma6, Sigma7):
    # physical regime takes the LOWER Σ
    return np.minimum(Sigma6, Sigma7)


# ================================================================
# D. MIDPLANE TEMPERATURE
# ================================================================

def midplane_temperature(R, Rp, Mp, Mdot, Sigma, kappaR, T_ext):
    A = (9*G*Mp*Mdot*Sigma*kappaR)/(128*np.pi*sigmaB*R**3)
    f = (1 - np.sqrt(Rp/R))
    Tc = (A*f + T_ext**4)**0.25
    return Tc


# ================================================================
# E. OPTICAL DEPTH AND BRIGHTNESS TEMPERATURE
# ================================================================

def tau_mm(Sigma, lam_mm):
    kappa_mm = 0.034 * (0.87/lam_mm)
    return 0.5 * kappa_mm * Sigma, kappa_mm


def Tb_from_tau(R, tau, Tc, Teff, T_ext, kappaR, kappa_mm):
    # thin approx
    Tb_thin = 2 * tau * Tc

    # thick approx (Zhu eq. 8 adaptation)
    Tb_thick = ((3/8)*(kappaR/kappa_mm)*Teff**4 + T_ext**4)**0.25

    Tb = np.where(tau <= 0.5, Tb_thin, Tb_thick)
    return Tb


# ================================================================
# F. FLUX INTEGRATION
# ================================================================

def flux_mm(R, Tb, lam_mm, d_pc ,Sigma):
    # ================================================================
    # === 9. Flux Density (Rayleigh-Jeans Approximation) =============
    # ================================================================
    Rmid = 0.5 * (R[:-1] + R[1:])
    dR = np.diff(R)
    I_RJ = 2.0 * kB * Tb / (lam_mm * 0.1)**2                                 # [erg cm^-2 s^-1 Hz^-1 sr^-1]
    F_nu_mid = 0.5 * (I_RJ[:-1] + I_RJ[1:])
    F_nu_tot = np.sum(F_nu_mid * 2 * np.pi * Rmid * dR)                      # [erg s^-1 Hz^-1]

    d_cm = d_pc * 3.086e18                                                   # [cm]
    F_nu_obs = F_nu_tot / d_cm**2                                            # [erg s^-1 cm^-2 Hz^-1]
    F_microJy = F_nu_obs / 1e-29                                             # [µJy]

    # ================================================================
    # === 10. Equaate CPD mass to power law sigma to sigma ===========
    # ================================================================
    M_dust = np.trapz(2 * np.pi * R * Sigma, R) / M_jup                  # [Earth masses]
    return F_microJy, M_dust


# ================================================================
# G. MASTER FUNCTION
# ================================================================

def calculate_disk_properties_zhu(M_star=1.0, Mp=1.0, Mdot=1e-6, alpha=1e-3, rp=20,
                  lam_mm=1.3, T_ISM=10, Lstar=1.0, kappaR=10, mu=2.3,
                  d_pc=140, Rout=None, Nr=200):

    # convert
    M_star, Mp_cgs, Mdot_cgs, rp_cgs, Lstar_cgs,Rp_cgs, Lplanet = convert_inputs(
        M_star, Mp, Mdot, rp, Lstar
    )



    # outer radius
    if Rout is None:
        Rout_cgs = (1/3)*rp_cgs*(Mp_cgs/(3*M_star))**(1/3)
    else:
        Rout_cgs = Rout*AU

    # grid
    R = radial_grid(Rp_cgs, Rout_cgs, Nr)

    # heating
    T_ext, T_irr, T_irr_p, T_irr_star = irradiation_temperature(R, Rp_cgs, Mp_cgs, Mdot_cgs,
                                                               Lplanet, Lstar_cgs, rp_cgs, T_ISM)
    Teff  = effective_temperature(R, Rp_cgs, Mp_cgs, Mdot_cgs)

    # Σ
    Sigma6 = sigma_viscous(R, Rp_cgs, Mp_cgs, Mdot_cgs, alpha, kappaR, mu)
    Sigma7 = sigma_irradiated(R, Mp_cgs, Mdot_cgs, alpha, mu, T_ext)
    Sigma  = sigma_combined(Sigma6, Sigma7)

    # Tc
    Tc = midplane_temperature(R, Rp_cgs, Mp_cgs, Mdot_cgs, Sigma, kappaR, T_ext)

    # τ, Tb
    tau, kappa_mm = tau_mm(Sigma, lam_mm)
    Tb = Tb_from_tau(R, tau, Tc, Teff, T_ext, kappaR, kappa_mm)

    # flux
    F_microJy, M_dust = flux_mm(R, Tb, lam_mm, d_pc,Sigma)

    return {
        "R": R,
        "R_out": Rout,
        "R_AU": R / AU,
        "T_ext": T_ext,
        "Teff": Teff,
        "T_c": Tc,
        "Tb": Tb,
        "Sigma": Sigma,
        "tau_mm": tau,
        "F_nu_tot": F_microJy,
        "M_dust": M_dust,
        "T_irr": T_irr,
        "T_irr_p": T_irr_p,
        "T_irr_star": T_irr_star,
    }

# ================================================================
# The plotting function
# ================================================================

def plot_zhu_Mp_Mdot_flux(
    Mp_grid, Mdot_grid, Flux_vals,
    disk_arr, rp, alpha,
    target_flux_arr,
    ax=None
):
    import numpy as np
    import matplotlib.pyplot as plt

    logMp = np.log10(Mp_grid)
    logMdot = np.log10(Mdot_grid)
    LOGMP, LOGMDOT = np.meshgrid(logMp, logMdot)
    Q = Mp_grid[None, :] * Mdot_grid[:, None]
    logQ = np.log10(Q)
    min_flux = min(target_flux_arr)
    Flux_masked = np.where(Flux_vals >= min_flux, Flux_vals, np.nan)
    logFlux = np.log10(Flux_masked)

    # 1. Create axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None

    # 2. Main contourf
    cf = ax.contourf(
        LOGMP, LOGMDOT, logFlux,
        levels=8,
        cmap="Spectral",
        extend="both"
    )
    if fig is not None:
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(r"$\log_{10}(F_\nu\, [\mu{\rm Jy}])$", fontsize=12)

    # 3. Q contours
    Q_levels = np.linspace(logQ.min()+0.5, logQ.max()-0.5, 5)
    cq = ax.contour(
        LOGMP, LOGMDOT, logQ,
        levels=Q_levels,
        colors="cyan",
        linestyles="dashed",
        linewidths=1.2
    )
    ax.clabel(cq, fmt=lambda x: f"Q={10**x:.1e}", fontsize=9)

    # 4. Sigma detection contours
    sigma_ujy = target_flux_arr[0]
    sigma_levels = [1, 2, 3, 5]
    flux_levels = [n * sigma_ujy for n in sigma_levels]
    cs = ax.contour(
        LOGMP, LOGMDOT, Flux_vals,
        levels=flux_levels,
        colors="white",
        linestyles="dashed",
        linewidths=1.6
    )
    ax.clabel(cs, fmt={lvl: f"{n}σ" for lvl, n in zip(flux_levels, sigma_levels)}, fontsize=10)

    # 5. Axis formatting
    ax.set_xlabel(r"$\log_{10}(M_p/M_{\rm Jup})$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\dot{M}\,[M_{\rm Jup}\,{\rm yr}^{-1}])$", fontsize=12)
    ax.set_title(
        rf"Zhu Model: {disk_arr[0]} at $r_p={rp}$ AU, $\alpha={alpha}$",
        fontsize=14
    )

    # 6. Regime boundaries
    Rfrac = 0.3
    M_star_jup = disk_arr[1] * 1047.56
    R_matrix = np.zeros_like(Flux_vals)
    for i, Mp in enumerate(Mp_grid):
        RHill = rp * (Mp / (3 * M_star_jup))**(1/3)
        Rcpd = Rfrac * RHill * AU
        R_matrix[i,:] = Rcpd

    T_visc = np.zeros_like(Flux_vals)
    T_pirr = np.zeros_like(Flux_vals)
    T_sirr = np.zeros_like(Flux_vals)
    for i, Mp in enumerate(Mp_grid):
        for j, Mdot in enumerate(Mdot_grid):
            R = R_matrix[i,j]
            Lp = Lp_from_Mp(Mp) * L_sun
            T_visc[i,j] = (3*G*(Mp*M_jup)*(Mdot*M_jup/sec_per_yr) /
                            (8*np.pi*sigmaB*R**3))**0.25
            T_pirr[i,j] = (0.1 * Lp / (4*np.pi*sigmaB*R**2))**0.25
            Lstar = disk_arr[3] * L_sun
            T_sirr[i,j] = (0.02 * Lstar / (8*np.pi*sigmaB*(rp*AU)**2))**0.25

    regime_index = np.argmax(np.stack([T_visc, T_pirr, T_sirr]), axis=0)
    logMp_grid, logMdot_grid = np.meshgrid(logMp, logMdot)
    ax.contour(
        logMp_grid, logMdot_grid, regime_index.T,
        levels=[0.5, 1.5], colors="pink", linewidths=1.5
    )
    ax.text(-0.8, -5.2, "viscous heating", color="pink", fontsize=12)
    ax.text( 0.4, -8.0, "planet irradiation", color="pink", fontsize=12)
    ax.text(-1.8, -7.5, "stellar irradiation", color="pink", fontsize=12)

    if fig is not None:
        plt.tight_layout()
        plt.show()



# ============================
# the compute flux function
# ===========================

def compute_flux_map_zhu_MpMdot(
    disk_arr,
    rp=165,
    Rout_frac=0.3,
    lam_mm=1.25,
    alpha=1e-3,
    Mp_min=-2, Mp_max=1, n_Mp=40,
    Mdot_min=-9, Mdot_max=-4, n_Mdot=40,
    verbose=True
):
    """
    Compute Zhu flux map in the (Mp, Mdot) plane
    with a fixed Rout/RH = constant.

    Returns:
        Mp_grid (n_Mp)
        Mdot_grid (n_Mdot)
        Flux_map (n_Mdot, n_Mp)     ← flux at each (Mdot, Mp)
        Q_map = Mp * Mdot
    """
    Mp_grid = np.logspace(Mp_min, Mp_max, n_Mp)
    Mdot_grid = np.logspace(Mdot_min, Mdot_max, n_Mdot)

    Flux_map = np.zeros((n_Mdot, n_Mp))
    Q_map = np.zeros_like(Flux_map)

    M_star_jup = disk_arr[1] * 1047.56

    for j, Mp in enumerate(Mp_grid):
        RHill = rp * (Mp / (3 * M_star_jup))**(1/3)
        Rout = Rout_frac * RHill

        for i, Mdot in enumerate(Mdot_grid):

            out = calculate_disk_properties_zhu(
                M_star=disk_arr[1],
                Mp=Mp,
                Mdot=Mdot,
                alpha=alpha,
                rp=rp,
                d_pc=disk_arr[2],
                lam_mm=lam_mm,
                T_ISM=10.0,
                Lstar=disk_arr[3],
                Rout=Rout
            )

            Flux_map[i,j] = out["F_nu_tot"]
            Q_map[i,j] = Mp * Mdot

            if verbose and (i % 10 == 0 and j % 10 == 0):
                print(f"Mp={Mp:.3f}, Mdot={Mdot:.2e}, Flux={out['F_nu_tot']:.3f}")

    return Mp_grid, Mdot_grid, Flux_map, Q_map
