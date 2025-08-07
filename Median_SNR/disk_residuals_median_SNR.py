# Load the residual fits file for each disk and store them in class then in object  

# --------------------------------------------------------
# Define a class (stores functions) for handling each disk
# --------------------------------------------------------

import os  
import numpy as np  
import re  # Python’s regular expressions module to extract numbers from filenames
from astropy.io import fits
import pandas as pd
from gofish import imagecube
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from matplotlib import cm, colors
import matplotlib.ticker as mtick

# Source detection
from photutils.segmentation import detect_sources, deblend_sources
from photutils.segmentation import SourceCatalog
from astropy.io import ascii  # ascii for saving source properties

class DiskResiduals_Median_SNR:

    #---------------------
    # Initialization
    #---------------------

    def __init__(self, name, path, geom_file):
        """
        Initializes a DiskResiduals object for one disk.
        - name: Disk name (e.g., 'AA_Tau')
        - path: Path to its residuals folder
        - geom_file: Full path to geometry .txt file
        - distance_pc: Distance to the disk in parsecs, used for unit conversion
        """
        self.rms_noise_FullFOV = None  # RMS noise for Full FOV images, to be set later
        self.name = name
        self.path = path
        self.inc, self.PA, self.center = self._load_geometry(geom_file)
        self.residuals = {}  # Dict to store {Briggs index value: FITS data}
        self.clean_images = {}   # Dict to store {Briggs index value: FITS data} for CLEAN images
        self.clean_profile = None  # Dict to store {Briggs index value: profile data}, currently None
        self.sigma_masks = {}
        self.snr_maps = {}  # Dict to store SNR maps for each robust value
        self.snr_map_FullFOV = {}

    #---------------------
    # Data Loading Methods
    #---------------------

    def _load_geometry(self, filepath):
        """
        Reads a galario geometry .txt file and returns (inc, PA, (dRA, dDec)).
        """
        with open(filepath, "r") as f:
            lines = [line for line in f.readlines() if not line.startswith("#")]  # Skip comment lines
        best_fit = [float(x) for x in lines[0].split()]  # Extract numbers from the first row of value
        inc, PA, dRA, dDec = best_fit  # Unpack the values ,excluding the rows below corresponding to errors
        return inc, PA, (dRA, dDec)   
    
    def load_disksize(self, radii_file):
        """Load R90 (disk size) and its errors from file, store as self.disksize (dict, arcsec)."""
        arr = np.loadtxt(radii_file, comments="#")
        self.disksize = {
            "R90": arr[1, 1],           # median R90 in arcsec
            "R90_err_low": arr[4, 1],   # lower error in arcsec
            "R90_err_high": arr[5, 1]   # upper error in arcsec
        }


    def load_ringgap(self, ringgap_path):
        """
        Load ring/gap data from a text file.
        File format (10 columns):
        Radial_location(au), Radial_location(arcsec), Flag(0=gaps,1=rings), 
        Width(au), Width(arcsec), Gap_depth, R_in(au), R_in(arcsec), R_out(au), R_out(arcsec)
        """
        arr = np.loadtxt(ringgap_path, comments="#")
        

        # If the array is 1D, convert it to 2D with one row so we can index it consistently
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        
        # Store the full array for use in plot_profiles
        self.ringgap = arr
        
        # Also create a more accessible dictionary format
        self.ringgap_info = {
            "radius_au": arr[:, 0],        # Column 0: radius in AU
            "radius_arcsec": arr[:, 1],    # Column 1: radius in arcsec
            "flag": arr[:, 2].astype(int), # Column 2: 0=gap, 1=ring
            "width_au": arr[:, 3],         # Column 3: width in AU
            "width_arcsec": arr[:, 4],     # Column 4: width in arcsec
            "gap_depth": arr[:, 5],        # Column 5: gap depth (NaN for rings)
            "r_in_au": arr[:, 6],          # Column 6: inner radius in AU
            "r_in_arcsec": arr[:, 7],      # Column 7: inner radius in arcsec
            "r_out_au": arr[:, 8],         # Column 8: outer radius in AU
            "r_out_arcsec": arr[:, 9]      # Column 9: outer radius in arcsec
        }  

    def load_residuals(self):
        """
        Load all .fits residuals in the folder and store them by Briggs Index.
        """
        for fname in os.listdir(self.path):
            if fname.endswith(".fits"):
                match = re.search(r"robust([-\d.]+)", fname)
                briggs_index = match.group(1) if match else "unknown"
                full_path = os.path.join(self.path, fname)
                with fits.open(full_path) as hdul:
                    self.residuals[briggs_index] = hdul[0].data
                    
    def _rkey(self, val):            # normalize "2" -> "2.0"
        return f"{float(val):.1f}"

    def has_robust(self, robust_val_str):
            """Return True if we can operate on this robust value (file exists or data loaded)."""
            r = f"{float(robust_val_str):.1f}"
            # consider already-loaded residuals or actual files present
            if r in self.residuals:
                return True
            fname = f"{self.name}_continuum_resid_robust{r}.image.fits"
            return os.path.exists(os.path.join(self.path, fname))
    
    def note_missing(self, full_path):
        msg = f"[MISSING] {self.name}: {os.path.basename(full_path)}"
        self._missing.append(msg)
        print(msg)

    def missing_report(self):
        """Return list of missing-file messages collected during the run."""
        return list(self._missing)
    


    def load_clean_images(self, clean_path):
        """
        Load all .fits CLEAN images in the specified path and store them by Briggs Index.
        """
        for fname in os.listdir(clean_path):
            if fname.endswith(".fits"):
                match = re.search(r"robust([-\d.]+)", fname)
                robust = match.group(1) if match else "unknown"
                full_path = os.path.join(clean_path, fname)
                with fits.open(full_path) as hdul:
                    self.clean_images[robust] = hdul[0].data
    
    def load_clean_profile(self, profile_path):
        """ Load the CLEAN profile from a text file.
        """
        arr = np.loadtxt(profile_path, comments="#")
        self.clean_profile = {
        "radius_au": arr[:, 1],  # column 1 is radius in au
        "intensity_Jy_sr": arr[:, 6],  # column 6 is intensity in Jy/sr
        "d_intensity_Jy_sr": arr[:, 7]
        }

    #---------------------
    # ImageCube Methods
    #---------------------

    def get_cube(self, robust_val, FOV=None, cube_type="residual", use_full_fov=False):
        r = self._rkey(robust_val)                      # normalize
        if cube_type == "residual":
            fname = f"{self.name}_continuum_resid_robust{r}.image.fits"
            folder = self.path
            full_path = os.path.join(folder, fname)
            if not os.path.exists(full_path):
                raise FileNotFoundError(full_path)
            return imagecube(full_path, FOV=FOV)
        elif cube_type == "clean":
            fname1 = f"{self.name}_continuum_data_robust{r}.image.fits"
            fname2 = f"{self.name}_continuum_data_robust{r}_FullFOV.image.fits"
            folder = self.path.replace("frank_residuals", "data")
            full_path1 = os.path.join(folder, fname1)
            full_path2 = os.path.join(folder, fname2)
            if use_full_fov:
                if os.path.exists(full_path2):
                    return imagecube(full_path2, FOV=FOV)
                else:
                    print(f"[WARN] {self.name}: Full FOV file not found: {full_path2}")
            else:
                if os.path.exists(full_path1):
                    return imagecube(full_path1, FOV=FOV)
                elif os.path.exists(full_path2):  # Fallback if main not found
                    print(f"[INFO] {self.name}: Default CLEAN file not found, using FullFOV instead.")
                    return imagecube(full_path2, FOV=FOV)
            print(f"WARNING: File not found: {full_path1} or {full_path2}")
            return None
        else:
            raise ValueError("cube_type must be 'residual' or 'clean'")
                    
    def plot_profiles(self, robust_val="1.0", FOV=10.0, radius_unit="arcsec"):
        """
        Plot CLEAN and residual radial profiles overlaid, with radius in arcsec atm.
        radius_unit: "arcsec" (default from GoFish), or "au"
        """
        r = self._rkey(robust_val)
        if not self.has_robust(r):
            print(f"[WARN] {self.name}: robust {r} not available — skipping plot.")
            return
        # Get cubes
        cube_clean = self.get_cube(r, FOV=FOV, cube_type="clean")
        cube_resid = self.get_cube(r, FOV=FOV, cube_type="residual")

        
        # Get radial profiles (x in arcsec , y in Jy/beam, dy in Jy/beam)
        # Assume correlated noise for the radial profile 
        x_cl, y_cl, dy_cl = cube_clean.radial_profile(
            inc=self.inc, PA=self.PA, unit='Jy/beam',assume_correlated=True, use_mad=True
        )
        x_res, y_res, dy_res = cube_resid.radial_profile(
            inc=self.inc, PA=self.PA, unit='Jy/beam', assume_correlated=True, use_mad=True
        )

        # Convert radius unit to au if requested
        if radius_unit == "au":
            x_cl = x_cl * self.distance_pc
            x_res = x_res * self.distance_pc
            xlabel = "Radius (au)"
        # Convert gaps/rings to au
            gap_unit_factor = self.distance_pc
        else:
            xlabel = "Radius (arcsec)"
            gap_unit_factor = 1.0

        # Create the plot
        fig, ax = plt.subplots(constrained_layout=True, figsize = (12, 6))


        # Plot the CLEAN and residual profiles, with error bars

        # CLEAN 
        ax.plot(x_cl, y_cl, color='gray', linewidth=2, label='CLEAN')
        ax.errorbar(x_cl, y_cl, dy_cl, fmt='none', ecolor='gray', alpha=0.4, capsize=2)

        # Residual
        ax.plot(x_res, dy_res, color='crimson', linewidth=2, label='Residual dy')
        #ax.errorbar(x_res, y_res, dy_res, fmt='none', ecolor='crimson', alpha=0.4 ,capsize=2)


        # Line for R90 disk size
        # hasattr(self, "disksize") checks if disksize is loaded
        # Convert to correct unit

        if hasattr(self, "disksize"):  
            R90 = self.disksize["R90"] * gap_unit_factor   
            err_low = self.disksize["R90_err_low"] * gap_unit_factor
            err_high = self.disksize["R90_err_high"] * gap_unit_factor
            ax.axvline(R90, color='k', linestyle='--', label='R90')
            ax.axvspan(R90 - err_low, R90 + err_high, color='k', alpha=0.15, label='R90 error')


        # Band for rings and gaps
        #


        if hasattr(self, "ringgap") and self.ringgap is not None and self.ringgap.size > 0:
            ringgap_arr = self.ringgap 

            # If ringgap_arr is a 1D array, convert to 2D with one row 
            if ringgap_arr.ndim == 1:     
                ringgap_arr = ringgap_arr[np.newaxis, :]

            # Flags to track if labels have been added
            gap_label_added = False
            ring_label_added = False

            # Each row is [Radial_location(au)	Radial_location(arcsec)	Flag(0 for gaps, 1 for rings)	Width(au)	Width(arcsec)	Gap_depth	R_in(au)	R_in(arcsec)	R_out(au)	R_out(arcsec)
            for row in ringgap_arr:  
                # Use data that's already in the correct units
                if radius_unit == "au":
                    rad = row[0]    # Column 0: radius in AU
                    width = row[3] if not np.isnan(row[3]) else None  # Column 3: width in AU
                else:
                    rad = row[1]    # Column 1: radius in arcsec  
                    width = row[4] if not np.isnan(row[4]) else None  # Column 4: width in arcsec

                flag = int(row[2])

                color = '#b9fbc0' if flag == 0 else '#cdb4fe'  # sage green for gaps, lavender for rings
                
                if flag == 0:   #
                    label = 'Gap' if not gap_label_added else None
                    gap_label_added = True
                else:
                    label = 'Ring' if not ring_label_added else None
                    ring_label_added = True
                ax.axvline(rad, color=color, linestyle=':', alpha=1.0, label=label)
                if width is not None and width > 0:
                    ax.axvspan(rad - width/2, rad + width/2, color=color, alpha=0.2)
                


        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Intensity (Jy/beam)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)  # box outside the plot
        ax.set_title(f"{self.name} robust={r}")
        plt.show()
    

    #------------------------------
    # Standard Deviation Methods
    #------------------------------

    def create_sigma_mask(self, robust_val="1.0", scale_factor=1.0, save_fits=False):
        """
        Create a 2D sigma mask from the radial profile standard deviation.
        
        Parameters:
        - robust_val: Briggs robust parameter (string)
        - scale_factor: Multiply sigma by this factor (e.g., 3.0 for 3-sigma)
        - save_fits: Whether to save the mask as a FITS file
        
        Returns:
        - sigma_2d: 2D array with sigma values for each pixel
        - radial_profile: tuple (x, y, dy) from the radial profile
        """

        r = self._rkey(robust_val)
        if not self.has_robust(r):
            print(f"[WARN] {self.name}: robust {r} not available — skipping sigma.")
            return None, None

        # Load the residual cube
        cube = self.get_cube(r, cube_type="residual")
        
        
        # Get radial profile with assume_correlated=False 
        # so that dy is simple the standard deviation per bin

        x, y, dy = cube.radial_profile(
            inc=self.inc, 
            PA=self.PA, 
            unit='Jy/beam', 
            assume_correlated=False,
            use_mad=True  # Use median absolute deviation for robust estimation

        )
        
        # Save files locally in organized folders
        output_base = "Disk_Residual_Profile_Median_SNR"
        disk_output_dir = os.path.join(output_base, self.name)
        os.makedirs(disk_output_dir, exist_ok=True)
        
        profile_filename = os.path.join(disk_output_dir, f"{self.name}_residual_radial_profile_robust{r}.txt")

        np.savetxt(
            profile_filename,
            np.column_stack([x, y, dy]),
            header="radius [arcsec] intensity [Jy/beam] standard deviation [Jy/beam]"
        )
        
        # Get 2D radius map
        # cube.disk_coords return a tuple of (rmap, theta_map, zmap)
        # A 2D array where each pixel at (x, y) is assigned to a radial value.
        rmap = cube.disk_coords(inc=self.inc, PA=self.PA)[0]
        

        # Get the radial bin edges in order to assign each pixel to a bin with a range
        # rvals: radial bin centers
        # cube.radial_sampling returns the radial bin edges and centers
        rbins, _ = cube.radial_sampling(rvals=x)
        
        # Assign each pixel to a bin
        # np.digitize returns the indices of the bins to which each value in rmap belongs
    
        bin_index = np.digitize(rmap, rbins) - 1  # 0-based , means from column[1] to column[0] as the first bin

        # Fill 2D array with sigma values
        sigma_2d = np.zeros_like(rmap) # Create an empty array with the same shape as rmap
        for i in range(len(dy)):
            sigma_2d[bin_index == i] = dy[i] * scale_factor  # Assign the sigma value to the corresponding pixels

        # Store in object
        self.sigma_masks[r] = sigma_2d

        # Save as FITS if requested into another subfolder


        if save_fits:
            fits_filename = f"{self.name}_sigma_mask_robust{r}.fits"
            if scale_factor != 1.0:
                fits_filename = f"{self.name}_sigma_mask_{scale_factor}sigma_robust{r}.fits"

            fits_path = os.path.join(disk_output_dir, fits_filename)
            fits.writeto(fits_path, sigma_2d, cube.header, overwrite=True)  # sigma_2d is 2D array, cube.header is the header from the residual cube
            print(f"Saved sigma mask: {fits_path}")
        

        # Store the sigma mask in the object
        self.sigma_mask = sigma_2d


        return sigma_2d, (x, y, dy)






    def plot_sigma_comparison(self, robust_val="1.0", scale_factor=1.0):
        """
        Plot original residual image alongside the sigma mask.
        """
        # Create sigma mask
    
        sigma_2d, _ = self.create_sigma_mask(robust_val, scale_factor, save_fits=False)
        
        # Load original data
        cube = self.get_cube(robust_val, cube_type="residual")
        orig_data = cube.data
        
        # Plot side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        
        im0 = axes[0].imshow(orig_data, origin='lower', cmap='inferno')
        axes[0].set_title(f"{self.name} Original Residual (robust={robust_val})")
        plt.colorbar(im0, ax=axes[0], label='Jy/beam')
        
        im1 = axes[1].imshow(sigma_2d, origin='lower', cmap='viridis')
        axes[1].set_title(f"{self.name} {scale_factor}σ Mask")
        plt.colorbar(im1, ax=axes[1], label=f'{scale_factor}σ (Jy/beam)')

        
        plt.show()


    # ┌──────────────────────────────────────────┐
    # │    Planet formation signature extraction │
    # └──────────────────────────────────────────┘

    #---------------------------------------------
    # Create SNR maps for disk residuals
    #---------------------------------------------


    def create_snr_map(self, robust_val="1.0"):
        """
        Create SNR map for THIS disk and store it in the object.
        """
        import warnings

        # Normalize key
        r = self._rkey(robust_val)
        if not self.has_robust(r):
            print(f"[WARN] {self.name}: robust {r} not available — skipping SNR.")
            return None

        if r in self.snr_maps:
            return self.snr_maps[r]

        try:
            sigma_2d = self.sigma_masks[r]
            residual_data = self.residuals[r]

            residual_data = np.squeeze(residual_data) 

            print(f"  {self.name}: Sigma shape: {sigma_2d.shape}, Residual shape: {residual_data.shape}")

            if sigma_2d.shape != residual_data.shape:
                raise ValueError(f"Dimension mismatch: sigma {sigma_2d.shape} vs residual {residual_data.shape}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                snr_map = residual_data / sigma_2d
                snr_map[sigma_2d == 0] = 0
                snr_map[~np.isfinite(snr_map)] = 0

            self.snr_maps[r] = snr_map
            return snr_map

        except Exception as e:
            print(f"  Error creating SNR map for {self.name}: {e}")
            return None
        
    
    #---------------------------------------------
    # Detect high SNR regions 
    #---------------------------------------------

    def catalogs_exist(disk_name, rkey, base="Disk_Residual_Profile_Median_SNR"):
        p3 = os.path.join(base, disk_name, f"robust_{rkey}",
                        f"source_catalog_{disk_name}_robust{rkey}_thresh3p0.txt")
        p5 = os.path.join(base, disk_name, f"robust_{rkey}",
                        f"source_catalog_{disk_name}_robust{rkey}_thresh5p0.txt")
        return os.path.exists(p3) and os.path.exists(p5)

    def source_detection(self, snr_map, robust_val, threshold=5.0, npixels=1, connectivity=4, overwrite=False):
        """
        Detect high-SNR sources and save a catalog.
        Skips work if the output file already exists (unless overwrite=True).
        """

        rkey = self._rkey(robust_val)
        if not self.has_robust(rkey):
            print(f"[WARN] {self.name}: robust {rkey} not available — skipping detection.")
            return None
        thr_str = str(threshold).replace('.', 'p')

        # Output path
        output_base = "Disk_Residual_Profile_Median_SNR"
        disk_output_dir = os.path.join(output_base, self.name, f"robust_{rkey}")
        os.makedirs(disk_output_dir, exist_ok=True)
        filename = os.path.join(
            disk_output_dir,
            f"source_catalog_{self.name}_robust{rkey}_thresh{thr_str}.txt"
        )

        # Skip if already done
        if (not overwrite) and os.path.exists(filename):
            print(f"  Catalog exists, skipping: {filename}")
            return filename

        # Pixel scale (info)
        cube = self.get_cube(rkey, cube_type="residual")
        pixel_scale_arcsec = abs(cube.header['CDELT1']) * 3600
        self.pixel_scale_au = pixel_scale_arcsec * self.distance_pc
        print(f"  Pixel scale: {self.pixel_scale_au:.1f} AU/pixel")

        # Detect
        segm = detect_sources(snr_map, threshold, npixels=npixels, connectivity=connectivity)
        if segm is None:
            print(f"  No sources detected above {threshold}σ")
            # Optional: create an empty file so future runs also skip
            # ascii.write(Table(), filename, format='commented_header', overwrite=True)
            return filename

        segm = deblend_sources(snr_map, segm, npixels=npixels, connectivity=connectivity)
        print(f"  Detected {segm.nlabels} sources")

        catalog = SourceCatalog(snr_map, segm).to_table()
        keep = [c for c in ['id','xcentroid','ycentroid','area','max_value','sum'] if c in catalog.colnames]
        catalog = catalog[keep]

        # Radii in AU
        rmap = cube.disk_coords(inc=self.inc, PA=self.PA)[0]
        radius_au = []
        for i in range(len(catalog)):
            x_pix = int(round(catalog['xcentroid'][i]))
            y_pix = int(round(catalog['ycentroid'][i]))
            radius_au.append(float(rmap[y_pix, x_pix] * self.distance_pc))
        catalog['radius_au'] = radius_au

        ascii.write(catalog, filename, format='commented_header', overwrite=True)
        print(f"  Saved source catalog to {filename}")
        return filename




    def complete_source_detection_summary(self):
        """
        Summarizes the source detection results from catalog files for all threshold values and robust values.

        Returns:
        - summary_df: DataFrame containing disk names and number of sources detected.
        """
        rows = []
        robust_vals = ["-2.0", "-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5", "2.0"]
        thresholds = [3, 5]

        base_path = "Disk_Residual_Profile_Median_SNR"  # Base directory for source catalogs
        for r in robust_vals:
            rkey = self._rkey(r)
            row = {"Robust": r}
            for t in thresholds:
                threshold_str = f"{t}p0"
                file_path = os.path.join(
                    base_path, self.name, f"robust_{rkey}",
                    f"source_catalog_{self.name}_robust{rkey}_thresh{threshold_str}.txt"
                )
        
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, comment=None, delim_whitespace=True, skiprows=1, names=[
                    'xcentroid', 'ycentroid', 'area', 'max_value', 'radius_au'
                    ])
                    row[f"{t}σ"] = len(df)
                else:
                    row[f"{t}σ"] = None  # File doesn't exist
            rows.append(row)

        summary_df = pd.DataFrame(rows)
        return summary_df
    

    
    def plot_snr_map_simple(self, robust_val, vmin=-6, vmax=6, show=True, use_full_fov=False):
        """
        Plot simplified SNR map with 3σ and 5σ contours and the R90 contour.
        Skips cleanly if required data/cubes are missing.
        """
        r = self._rkey(robust_val)

        # Ensure we have/compute an SNR map
        if use_full_fov:
            print(f"[INFO] {self.name}: Using full FOV for robust {r} SNR map.")
            snr_map = self.snr_map_FullFOV.get(r)
        else:
            print(f"[INFO] {self.name}: Using default FOV for robust {r} SNR map.")
            snr_map = self.snr_maps.get(r)

        if snr_map is None:
            snr_map = self.create_snr_map(r)
        if snr_map is None:
            print(f"[WARN] {self.name}: robust {r} SNR map unavailable — skipping plot.")
            return

        # Need residual cube for header (pixel scale) and disk_coords
        if use_full_fov:
            print(f"[INFO] {self.name}: Using full FOV clean cube for robust {r}.")
            cube = self.get_cube("2.0", cube_type="clean", use_full_fov=True)
        else:
            cube = self.get_cube(r, cube_type="residual")
            if cube is None:
                print(f"[WARN] {self.name}: robust {r} residual cube missing — skipping plot.")
                return

        # Binary masks for contours
        mask_3sigma = snr_map >= 3.0
        mask_5sigma = snr_map >= 5.0

        # Pixel scale (arcsec/pixel); abs in case CDELT1 < 0
        try:
            pixel_scale_arcsec = abs(float(cube.header.get('CDELT1'))) * 3600.0
        except Exception:
            pixel_scale_arcsec = None

        # R90 in arcsec and AU
        r90_arcsec = self.disksize["R90"]  
        r90_au = r90_arcsec * float(self.distance_pc)

        # Deprojected radial map (same units as r90_arcsec)
        rmap = cube.disk_coords(inc=self.inc, PA=self.PA)[0]

        # ---- Plot ----
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.figure(figsize=(8, 7))
        # purple-green colormap: PiYG, greyscale: Greys
        plt.imshow(snr_map, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
        cb = plt.colorbar()
        cb.set_label('SNR')

        # Contours
        plt.contour(mask_3sigma, levels=[0.5], colors='yellow', linewidths=1.5, linestyles='--')
        plt.contour(mask_5sigma, levels=[0.5], colors='Red', linewidths=4, linestyles='-')
        plt.contour(rmap, levels=[r90_arcsec], colors='orange', linewidths=2.5, linestyles='-')

        # Labels
        plt.title(f"{self.name} — SNR Contours (robust={r})")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")

        # Legend (manual)
        custom_lines = [
            Line2D([0], [0], color='brown', lw=2, linestyle='--', label='>3σ'),
            Line2D([0], [0], color='blue',  lw=2, linestyle='-',  label='>5σ'),
            Line2D([0], [0], color='orange',lw=2, linestyle='-',  label=f'R90 = {r90_au:.1f} AU'),
        ]
        plt.legend(handles=custom_lines, loc='upper right')

        if show:
            plt.show()


    def save_snr_map_as_fits(self, robust_val="2.0", output_dir="SNR_FITS_Maps", overwrite=False):
        """
        Save the 2D SNR map as a FITS file using the (trimmed) header from the residual map.
        Skips if SNR/cube/header/shape are missing or inconsistent. Returns output path or None.
        """
        # Normalize robust token consistently with the rest of your code
        rkey = self._rkey(robust_val) if hasattr(self, "_rkey") else f"{float(robust_val):.1f}"

        # Build output path to skip heavy work if file exists
        os.makedirs(output_dir, exist_ok=True)
        outname = os.path.join(output_dir, f"{self.name}_SNR_robust{rkey}.fits")

        # Early skip if file already exists
        if (not overwrite) and os.path.exists(outname):
            print(f"[SKIP] {self.name}: {outname} already exists.")
            return outname

        # Get SNR (build if needed)
        snr_map = self.snr_maps.get(rkey)
        if snr_map is None and hasattr(self, "create_snr_map"):
            snr_map = self.create_snr_map(rkey)
        if snr_map is None:
            print(f"[SKIP] {self.name}: robust {rkey} — no SNR map.")
            return None

        # Residual cube/header
        cube = self.get_cube(rkey, cube_type="residual")
        if cube is None or getattr(cube, "header", None) is None:
            print(f"[SKIP] {self.name}: robust {rkey} — no residual cube/header.")
            return None
        header = cube.header.copy()

        # Shape guard (FITS is [NAXIS2, NAXIS1])
        ny, nx = snr_map.shape
        if ("NAXIS1" in header and "NAXIS2" in header) and (ny, nx) != (header["NAXIS2"], header["NAXIS1"]):
            print(f"[SKIP] {self.name}: robust {rkey} — shape mismatch SNR {snr_map.shape} vs header ({header['NAXIS2']}, {header['NAXIS1']}).")
            return None

        # Trim to 2D: remove axis 3/4 + spectral/Stokes keys
        for i in (3, 4):
            for k in (f"NAXIS{i}", f"CTYPE{i}", f"CDELT{i}", f"CRPIX{i}", f"CRVAL{i}", f"CUNIT{i}"):
                if k in header: del header[k]
        for k in ("SPECSYS","RESTFRQ","RESTFREQ","RESTWAV","RESTWAVE","VELREF","STOKES","BTYPE"):
            if k in header: del header[k]
        header["NAXIS"]  = 2
        header["NAXIS1"] = int(nx)
        header["NAXIS2"] = int(ny)
        header["BUNIT"]  = "SNR"
        try:
            header.add_history("2D SNR map created from residual/sigma maps.")
            header.add_history(f"robust={rkey}")
        except Exception:
            pass  # some headers may not support HISTORY (very rare)

        # Write
        os.makedirs(output_dir, exist_ok=True)
        outname = os.path.join(output_dir, f"{self.name}_SNR_robust{rkey}.fits")
        fits.writeto(outname, np.asarray(snr_map, dtype=np.float32), header, overwrite=True)
        print(f"[OK]  {self.name}: saved {outname}")
        return outname
    

 

    def plot_residual_dy_from_files(self, robust_vals,
                                    base_folder="Disk_Residual_Profile_Median_SNR",
                                    radius_unit="arcsec", figsize=(8, 5)):

        disk_dir = Path(base_folder) / self.name
        files_found = 0

        # color map across robust values
        cmap = cm.get_cmap("tab10") 
        norm = colors.Normalize(vmin=0, vmax=max(1, len(robust_vals)-1))

        fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)

        for i, r in enumerate(robust_vals):
            p = disk_dir / f"{self.name}_residual_radial_profile_robust{r}.txt"
            if not p.exists():
                print(f"[WARN] missing robust {r}: {p.name}")
                continue

            data = np.loadtxt(p, comments="#")
            x = data[:, 0]
            dy = data[:, 2]

            if radius_unit == "au":
                x = x * self.distance_pc
                xlabel = "Radius (au)"
                 # Convert gaps/rings to au
                gap_unit_factor = self.distance_pc
            else:
                xlabel = "Radius (arcsec)"
                gap_unit_factor = 1.0

            

            color = cmap(norm(i))
            ax.plot(x, dy, lw=1.8, color=color, label=f"robust {r}")
            files_found += 1

        if files_found == 0:
            print(f"[WARN] no residual profiles found under {disk_dir}")
            plt.close(fig)
            return
        
        # Add R90 line
    
        
        R90 = self.disksize["R90"] * gap_unit_factor   
            
        ax.axvline(R90, color='k', linestyle='--', label='R90')
            
        

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mtick.LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_minor_locator(mtick.LogLocator(base=10, subs=np.arange(2,10)*0.1))
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, p: f"{v:.0e}"))

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Standard deviation dy (Jy/beam)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=9)
        ax.set_title(f"{self.name} — residual dy vs radius")
        # Add dotted grid
        ax.grid(which='both', linestyle=':', linewidth=0.5, color='gray', alpha=0.4)
        plt.show()

    def source_detection_outer_only(self,  robust_val, threshold=5.0, npixels=1, connectivity=4, overwrite=False):
        """
        Detect high-SNR sources only outside 2 × R90 and save a catalog.
        Skips work if the output file already exists (unless overwrite=True).
        """

        rkey = self._rkey(robust_val)
        if not self.has_robust(rkey):
            print(f"[WARN] {self.name}: robust {rkey} not available — skipping detection.")
            return None

        if not hasattr(self, "disksize") or "R90" not in self.disksize:
            print(f"[WARN] {self.name}: R90 not available — cannot mask inner region.")
            return None

        thr_str = str(threshold).replace('.', 'p')

        # Output path
        output_base = r"D:\CPD_MPIA\Median_SNR\Disk_Residual_Profile_Median_SNR"
        disk_output_dir = os.path.join(output_base, self.name, f"robust_{rkey}")
        os.makedirs(disk_output_dir, exist_ok=True)
        filename = os.path.join(
            disk_output_dir,
            f"source_catalog_{self.name}_robust{rkey}_thresh{thr_str}_outerOnly.txt"
        )

        # Skip if already done
        if (not overwrite) and os.path.exists(filename):
            print(f"  Catalog exists, skipping: {filename}")
            return filename

        # Pixel scale
        cube = cube = self.get_cube("2.0", cube_type="clean", use_full_fov=True)
        # Load CLEAN image (Full FOV)
        image_data = np.squeeze(cube.data)  # 2D CLEAN image
        rms = self.rms_noise_FullFOV

        # Compute simple SNR map
        snr_map = image_data / rms


        self.snr_map_FullFOV[rkey] = snr_map # Store for later use

        pixel_scale_arcsec = abs(cube.header['CDELT1']) * 3600
        self.pixel_scale_au = pixel_scale_arcsec * self.distance_pc
        print(f"  Pixel scale: {self.pixel_scale_au:.1f} AU/pixel")

        # Get radial map
        rmap = cube.disk_coords(inc=self.inc, PA=self.PA)[0]

        # Create mask: keep only r >= 2 × R90
        r90_arcsec = self.disksize["R90"]
        outer_mask = rmap >= (2.0 * r90_arcsec)

        # Apply mask: zero out inner region
        masked_snr_map = snr_map.copy()
        masked_snr_map[~outer_mask] = 0.0  

        # Detect sources
        segm = detect_sources(masked_snr_map, threshold, npixels=npixels, connectivity=connectivity)
        if segm is None:
            print(f"  No sources detected above {threshold}σ beyond 2×R90")
            return filename

        segm = deblend_sources(masked_snr_map, segm, npixels=npixels, connectivity=connectivity)
        print(f"  Detected {segm.nlabels} sources beyond 2×R90")

        catalog = SourceCatalog(masked_snr_map, segm).to_table()
        keep = [c for c in ['id', 'xcentroid', 'ycentroid', 'area', 'max_value', 'sum'] if c in catalog.colnames]
        catalog = catalog[keep]

        # Compute radii in AU
        radius_au = []
        for i in range(len(catalog)):
            x_pix = int(round(catalog['xcentroid'][i]))
            y_pix = int(round(catalog['ycentroid'][i]))
            radius_au.append(float(rmap[y_pix, x_pix] * self.distance_pc))
        catalog['radius_au'] = radius_au

        # Save
        ascii.write(catalog, filename, format='commented_header', overwrite=True)
        print(f"  Saved outer-only source catalog to {filename}")
        return filename