# import everything first:
import numpy as np
import astropy.constants as const
import astropy.units as u
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import pickle
from joblib import load

# Constants
G = 6.674e-8        # cm^3 g^-1 s^-2
sigmaB = 5.67e-5    # erg cm^-2 s^-1 K^-4
kB = 1.381e-16      # erg/K
L_sun = 3.828e33    # erg/s
R_p = 7.149e9       # Jupiter radius in cm
AU = 1.496e13       # cm
L_jup = 3.846e30  # erg/s

# Physical constants
M_sun = 1.989e33 
M_jup = 1.898e30
R_jup = 7.149e9
#AU = 1.496e13
sec_per_yr = 3.156e7
Rgas = 8.314e7  # erg/(mol K)


# ============================ 
# Load pre-calculated relations
# ============================

Lp_from_Mp = load('d:/CPD_MPIA/utils/Lp_from_Mp.pkl')
Rp_from_Mp = load('d:/CPD_MPIA/utils/Rp_from_Mp.pkl')


def B_nu(nu, T):
    """
    Planck function in frequency (erg s^-1 cm^-2 Hz^-1 sr^-1)
    nu: frequency in Hz
    T: temperature in K
    """
    h = 6.626e-27  # erg*s
    c = 2.998e10   # cm/s
    k = 1.381e-16  # erg/K
    
    return (2*h*nu**3 / c**2) / (np.exp(h*nu/(k*T)) - 1)

def calculate_disk_properties_Andrews(M_star = 1, Mp = 1, Mdot = 1.56, alpha = 1e-3, kappaR = 10 , Rin = 1, rp = 22, d_pc = 167, 
                             lam_nu=240 , 
                             Lstar = 1.0 , show_results = False, Rout=None, M_cpd = 0.01, inc = 16 , add_T = None):
    """
    Calculate disk temperature profiles and millimeter flux.
    
    Parameters:
    -----------
    M_star : float
        Stellar mass (M_sun)
    lam_nu : float
        Frequency (GHz)
    Mp : float
        Planet mass (M_jup)
    Mdot : float  
        Accretion rate (M_jup/Myr)
    alpha : float
        Viscosity parameter
    kappaR : float
        Rosseland opacity (cm^2/g)
    Rin : float
        Inner radius (Rjup)
    Rp : float
        distance of planet from star (au)  
    d_pc : float
        Distance (pc)
    mode : str
        Irradiation mode: "b", "no_b", or "planet"
    T_ISM : float
        Interstellar medium temperature (K)
    Lplanet : float
        Planet luminosity (L_sun)
    

    Lstar : float
        Stellar luminosity (L_sun)
        
    Returns:
    --------
    dict : Dictionary containing R, temperatures, Sigma, tau_mm, Tb, F_nu_tot
    """
    # Turn input values to cgs
    # Disk parameters
    M_star = M_star * M_sun
    Mp = Mp * M_jup
    Mdot = Mdot * M_jup / (sec_per_yr*1e6)
    Rin = Rin * R_jup
    rp = rp * AU # planet orbital radius in cm
    Rp = Rp_from_Mp(Mp / M_jup)*R_jup  # planet radius in cm

    # Turn the luminosites to erg/s
    Lstar = Lstar * L_sun
    
    # 1. Create radial grid between Rin and Rout

    

    if Rout is None:
        Rout = (1/3) * rp * (Mp / (3 * M_star))**(1/3)
    else:
        Rout = Rout * AU  # assume input Rout is in AU


    R = np.geomspace(Rp*1.01, Rout, 100)  # cm
    
    # 2. Calculate T(R)
    # Accretion irradiation
    
    T_irr = (3*G * Mp * Mdot / (8*np.pi*sigmaB*R**3) * (1 - np.sqrt(Rp/R)))**(1/4)  # K # RJUP CAN BE WRONG  NEED PLANET RADIUS
    
    #print(f'L_irr = {L_irr/L_sun:.2e} L_sun')

    # Planet irradiation
    Lplanet = Lp_from_Mp(Mp / M_jup) * L_sun  # erg/s
    T_irr_p = (0.1 * Lplanet  / (4 * np.pi * sigmaB * R**2))**0.25  # K  , R is coordinate in CPD frame

    # Star irradiation

    phi_flare = 0.02   # flaring angle of the host disk   (Huang et al. (2018b))

    T_irr_star = (phi_flare * Lstar / (8 * np.pi * sigmaB * rp**2))**0.25  # K

    T_ext= (T_irr**4 + T_irr_p**4 + T_irr_star**4 )**0.25  # K
    

    if add_T is not None:
        T_ext = (T_ext**4 + add_T**4)**0.25
        
    
    #---------------------------------------
    # -------------SIGMA--------------------
    # --------------------------------------
    
    gamma = 0.75
    M_cpd = M_cpd * M_jup  # CPD mass in grams


    Sigma0 = (2 - gamma) * M_cpd /  (2 * np.pi * Rout**gamma) * (Rout**(2 - gamma) - Rp**(2 - gamma))**(-1)
    Sigma = Sigma0 * (R/Rout)**(-gamma)




    #---------------------------------------
    #---------------Flux--------------------
    # --------------------------------------
      
 

    kappa_nu = 3.5*(lam_nu/345)  #cm^2/g

    tau_nu = kappa_nu * Sigma

    # Assume face-on disk: mu = cos(i) = 1

    # turn inc from degree to radian

    mu = np.cos(np.radians(inc))# i is the inclination angle in degree in table 2

    # Convert distance to cm
    pc_to_cm = 3.086e18  # cm/pc
    d_cm = d_pc * pc_to_cm

    # Integrand: B_nu(T) * (1 - exp(-tau/mu)) * r
    # Ignoring omega_nu (scattering) term as specified
    integrand = B_nu(lam_nu * 1e9, T_ext) * (1 - np.exp(-tau_nu / mu)) * R

    # Integrate using trapezoidal rule
    flux_integral = np.trapz(integrand, R)

    # Total observed flux at Earth
    F_cpd = (2 * np.pi * mu / d_cm**2) * flux_integral  # erg s^-1 cm^-2 Hz^-1

    # Convert to microJy
    F_microJy = F_cpd / 1e-29  # µJy

    
    return {
        'R': R,
        'Rout': Rout / AU,
        'R_AU': R / AU,
        'T_ext': T_ext,
        'Sigma': Sigma,
        'tau_nu_max': np.max(tau_nu),
        'F_nu_tot': F_microJy,
        'M_cpd': M_cpd / M_jup,
        'T_irr': T_irr,
        'T_irr_p': T_irr_p,
        'T_irr_star': T_irr_star
    }


def compute_flux_map(
    target_flux_arr,
    disk_arr,
    rp=165,
    alpha=1e-3,
    y_max=0.8,
    y_min=0.05,
    m_max = 1.25,
    m_min = -2,
    mcpd_min = -5.5,
    mcpd_max = -3,
    lam_nu=240,
    inc=16,
    mode="default",
    overwrite=False,
    save=True,
    verbose=True,
    n_jobs= 4
):
    """
    Compute flux and CPD mass grids for given disk parameters.
    Optionally caches results in /Andrews folder.

    Returns:
        Mp_grid, Rout_frac_grid, Flux_vals, M_cpd_chosen
    """
    import os
    import numpy as np
    import pandas as pd

    # === Optional caching ===
    save_dir = "Andrews"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{disk_arr[0]}_{rp}_alpha{alpha}_{mode}.csv"
    save_path = os.path.join(save_dir, fname)

    if save and os.path.exists(save_path) and not overwrite:
        if verbose:
            print(f"Loading cached results from {save_path}")
        df = pd.read_csv(save_path)
        Mp_grid = df['Mp_grid'].unique()
        Rout_frac_grid = df['Rout_frac_grid'].unique()
        Flux_vals = df['Flux_vals'].values.reshape(len(Rout_frac_grid), len(Mp_grid))
        M_cpd_chosen = df['Mcpd_vals'].values.reshape(len(Rout_frac_grid), len(Mp_grid))
        return Mp_grid, Rout_frac_grid, Flux_vals, M_cpd_chosen

    # === Computation ===
    Mp_grid = np.logspace(m_min, m_max, 100)
    M_cpd_grid = np.logspace(mcpd_min, mcpd_max, 80)
    Rout_frac_grid = np.linspace(y_min, y_max, 80)
    M_star_jup = disk_arr[1] * 1047.56

    Flux_vals = np.zeros((len(Rout_frac_grid), len(Mp_grid)))
    M_cpd_chosen = np.zeros_like(Flux_vals)

    # ================================================================
    # === 3. Define per-cell computation ==============================
    # ================================================================
    def compute_cell(i, j, frac, Mp):
        """Compute one (Rout/RH, Mp) grid cell."""
        try:
            # --- Hill radius & outer disk radius (in AU) ---
            RHill = rp * (Mp / (3 * M_star_jup)) ** (1/3)
            Rout = frac * RHill

            # --- Skip unphysical small disks ---
            Rp_planet = Rp_from_Mp(Mp)  # [Rjup]
            Rp_planet_AU = Rp_planet * R_jup / AU
            if Rout <= Rp_planet_AU:
                return (i, j, 0.0, 0.0)

            # --- Iterate through CPD masses until flux threshold reached ---
            for M_cpd in M_cpd_grid:
                out = calculate_disk_properties_Andrews(
                    M_star=disk_arr[1], Mp=Mp, alpha=alpha,
                    Rin=1, rp=rp, d_pc=disk_arr[2],
                     Lstar=disk_arr[3],
                    Rout=Rout, M_cpd=M_cpd, inc=inc, lam_nu=lam_nu
                )
                flux = out["F_nu_tot"]
                M_cpd = out["M_cpd"]
                

                # Stop at first mass exceeding detection threshold
                if flux >= target_flux_arr[0]:
                    if verbose and (i % 10 == 0 and j % 20 == 0):
                        print(f"[{i:02d},{j:02d}] Rout/RH={frac:.3f}, Mp={Mp:.3f} -> Fν={Flux_vals[i,j]:.2f} µJy")

                    return (i, j, flux, M_cpd)


                # No detection: return zeros
            return (i, j, 0.0, 0.0)
        
        
        except Exception as e:
            if verbose:
                print(f"[WARN] Cell (i={i}, j={j}) failed: {e}")
            return (i, j, 0.0, 0.0)


    # ================================================================
    # === 4. Run parallel computation ================================
    # ================================================================
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(compute_cell)(i, j, frac, Mp)
        for i, frac in enumerate(Rout_frac_grid)
        for j, Mp in enumerate(Mp_grid)
    )

    # ================================================================
    # === 5. Collect and reshape results ==============================
    # ================================================================
    Flux_vals = np.zeros((len(Rout_frac_grid), len(Mp_grid)))
    M_cpd_chosen = np.zeros_like(Flux_vals)

    for i, j, flux, M_cpd in results:
        Flux_vals[i, j] = flux
        M_cpd_chosen[i, j] = M_cpd

    # ================================================================
    # === 6. Summary and diagnostics =================================
    # ================================================================
    if verbose:
        n_nonzero = np.sum(Flux_vals > 0)
        print(f"[INFO] Non-zero flux points: {n_nonzero}/{Flux_vals.size}")
        if n_nonzero == 0:
            print("[WARN] All fluxes are zero — check units or disk model scaling.")

    # === Save ===
    if save:
        df = pd.DataFrame({
            'Mp_grid': np.tile(Mp_grid, len(Rout_frac_grid)),
            'Rout_frac_grid': np.repeat(Rout_frac_grid, len(Mp_grid)),
            'Flux_vals': Flux_vals.flatten(),
            'Mcpd_vals': M_cpd_chosen.flatten()
        })
        df.to_csv(save_path, index=False)
        if verbose:
            print(f"Saved results to {save_path}")

    return Mp_grid, Rout_frac_grid, Flux_vals, M_cpd_chosen

def plot_flux_map(Mp_grid, Rout_frac_grid, Flux_vals, M_cpd_chosen,
                  disk_arr, rp=165, alpha=1e-3, res_au = 7.4 ,
                  color='Spectral', y_max=0.8, y_min=0.05,x_min = -1.5, x_max =1,
                  sigma_ujy=None, det_sigma=None,
                  min=-5.5, max=-3.2,
                  ax=None):

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rcParams  # <--- ADDED for font consistency

    # ============= Create axis if needed ==================
    external_ax = True
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        external_ax = False

    # ============= Set aspect ratio for consistency ==================
    ax.set_aspect('auto')  # <--- ADDED for consistent subplot sizing

    # ============= Contourf map ==================
    levels = np.arange(min, max, 0.2)
    cf = ax.contourf(
        np.log10(Mp_grid),
        Rout_frac_grid,
        np.log10(M_cpd_chosen),
        levels=levels,
        cmap=color
    )

    # Add colorbar ONLY if we're not inside subplot
    if not external_ax:
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(r'$\log_{10}(M_{\rm CPD}/M_{\rm Jup})$', fontsize=rcParams['font.size'])  # <--- CHANGED from 11

    # ============= Beam-limited Rout/RH curve ==================
    M_star_jup = disk_arr[1] * 1047.56
    RHill_arr = rp * (Mp_grid / (3 * M_star_jup))**(1/3)
    Rout_frac_beam = res_au / (2 * RHill_arr)

    # ============= detection limit contours ==================
    ax.plot(np.log10(Mp_grid), Rout_frac_beam, color='purple', linestyle='-.', lw=3, label='_nolegend_')

    # ============= Labels ================

    ax.set_xlabel(r"$\log_{10}(M_p/M_{Jup})$", fontsize=rcParams['font.size'])  # <--- ADDED fontsize
    ax.set_ylabel(r"$R_{\rm cpd}/R_H$", fontsize=rcParams['font.size'])  # <--- ADDED fontsize
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.tick_params(labelsize=rcParams['font.size']*0.9)  # <--- ADDED for tick label size
    #ax.legend(fontsize=rcParams['font.size']*0.9)  # <--- CHANGED from 8

    # ============= Title only if standalone ================
    if not external_ax:
        ax.set_title(f"{disk_arr[0]} at $r_p$={rp:.1f} AU, α={alpha}", 
                     fontsize=rcParams['font.size'])  # <--- ADDED fontsize
        plt.tight_layout()
        plt.show()

    return cf   # <--- return the contourf object for shared colorbars

