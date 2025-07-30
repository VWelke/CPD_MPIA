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

# Source detection
from photutils.segmentation import detect_sources, deblend_sources
from photutils.segmentation import SourceCatalog
from astropy.io import ascii  # ascii for saving source properties

class DiskResiduals:

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
        self.name = name
        self.path = path
        self.inc, self.PA, self.center = self._load_geometry(geom_file)
        self.residuals = {}  # Dict to store {Briggs index value: FITS data}
        self.clean_images = {}   # Dict to store {Briggs index value: FITS data} for CLEAN images
        self.clean_profile = None  # Dict to store {Briggs index value: profile data}, currently None
        self.sigma_masks = {}
        self.snr_maps = {}  # Dict to store SNR maps for each robust value

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

    def get_cube(self, robust_val, FOV=None, cube_type="residual"):
        """
        Returns a GoFish ImageCube for the disk and robust index.
        cube_type: "residual" or "clean"
        """
        if cube_type == "residual":
            fname = f"{self.name}_continuum_resid_robust{robust_val}.image.fits"
            folder = self.path
        elif cube_type == "clean":
            # Try with and without _FullFOV
            fname1 = f"{self.name}_continuum_data_robust{robust_val}.image.fits"
            fname2 = f"{self.name}_continuum_data_robust{robust_val}_FullFOV.image.fits"
            folder = self.path.replace("frank_residuals", "data")
            full_path1 = os.path.join(folder, fname1)
            full_path2 = os.path.join(folder, fname2)
            if os.path.exists(full_path1):
                return imagecube(full_path1, FOV=FOV)
            elif os.path.exists(full_path2):
                return imagecube(full_path2, FOV=FOV)
            else:
                print(f"WARNING: File not found: {full_path1} or {full_path2}")
                return None
        else:
            raise ValueError("cube_type must be 'residual' or 'clean'")
        
        full_path = os.path.join(folder, fname)
        return imagecube(full_path, FOV=FOV)
                    
    def plot_profiles(self, robust_val="1.0", FOV=10.0, radius_unit="arcsec"):
        """
        Plot CLEAN and residual radial profiles overlaid, with radius in arcsec atm.
        radius_unit: "arcsec" (default from GoFish), or "au"
        """
        # Get cubes
        cube_clean = self.get_cube(robust_val, FOV=FOV, cube_type="clean")
        cube_resid = self.get_cube(robust_val, FOV=FOV, cube_type="residual")

        
        # Get radial profiles (x in arcsec , y in Jy/beam, dy in Jy/beam)
        # Assume correlated noise for the radial profile 
        x_cl, y_cl, dy_cl = cube_clean.radial_profile(
            inc=self.inc, PA=self.PA, unit='Jy/beam',assume_correlated=True
        )
        x_res, y_res, dy_res = cube_resid.radial_profile(
            inc=self.inc, PA=self.PA, unit='Jy/beam',assume_correlated=True
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
        ax.plot(x_res, y_res, color='crimson', linewidth=2, label='Residual')
        ax.errorbar(x_res, y_res, dy_res, fmt='none', ecolor='crimson', alpha=0.4 ,capsize=2)


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
        ax.set_title(f"{self.name} robust={robust_val}")
        plt.show()
    

    #------------------------------
    # Standard Deviation Methods
    #------------------------------

    def create_sigma_mask(self, robust_val="1.0", scale_factor=1.0, save_fits=True):
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

        # Load the residual cube
        cube = self.get_cube(robust_val, cube_type="residual")
        
        
        # Get radial profile with assume_correlated=False 
        # so that dy is simple the standard deviation per bin

        x, y, dy = cube.radial_profile(
            inc=self.inc, 
            PA=self.PA, 
            unit='Jy/beam', 
            assume_correlated=False
        )
        
        # Save files locally in organized folders
        output_base = "Disk_Residual_Profile"
        disk_output_dir = os.path.join(output_base, self.name)
        os.makedirs(disk_output_dir, exist_ok=True)
        
        profile_filename = os.path.join(disk_output_dir, f"{self.name}_residual_radial_profile_robust{robust_val}.txt")

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
        self.sigma_masks[robust_val] = sigma_2d

        # Save as FITS if requested into another subfolder


        if save_fits:
            fits_filename = f"{self.name}_sigma_mask_robust{robust_val}.fits"
            if scale_factor != 1.0:   
                fits_filename = f"{self.name}_sigma_mask_{scale_factor}sigma_robust{robust_val}.fits"
            
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
        robust_key = f"{float(robust_val):.1f}"

        if robust_key in self.snr_maps:
            return self.snr_maps[robust_key]

        try:
            sigma_2d = self.sigma_masks[robust_key]
            residual_data = self.residuals[robust_key]

            residual_data = np.squeeze(residual_data) 

            print(f"  {self.name}: Sigma shape: {sigma_2d.shape}, Residual shape: {residual_data.shape}")

            if sigma_2d.shape != residual_data.shape:
                raise ValueError(f"Dimension mismatch: sigma {sigma_2d.shape} vs residual {residual_data.shape}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                snr_map = residual_data / sigma_2d
                snr_map[sigma_2d == 0] = 0
                snr_map[~np.isfinite(snr_map)] = 0

            self.snr_maps[robust_key] = snr_map
            return snr_map

        except Exception as e:
            print(f"  Error creating SNR map for {self.name}: {e}")
            return None
        
    
    #---------------------------------------------
    # Detect high SNR regions 
    #---------------------------------------------

    def source_detection(self, snr_map, robust_val, threshold=5.0, npixels=1, connectivity=4):
        """
        Detect high SNR sources. Save detected properties to a text file.
        """
        threshold_str = str(threshold).replace('.', 'p')

        # Get pixel scale
        cube = self.get_cube(robust_val, cube_type="residual")
        pixel_scale_arcsec = abs(cube.header['CDELT1']) * 3600  # Convert deg to arcsec
        pixel_scale_au = pixel_scale_arcsec * self.distance_pc

        self.pixel_scale_au = pixel_scale_au

        print(f"  Pixel scale: {pixel_scale_au:.1f} AU/pixel")

        # Detect sources above threshold
        segm = detect_sources(snr_map, threshold, npixels=npixels, connectivity=connectivity)

        if segm is not None:
            segm_deblend = deblend_sources(snr_map, segm, npixels=npixels, connectivity=connectivity)
            print(f"  Detected {segm_deblend.nlabels} sources")
        else:
            segm_deblend = None
            print(f"  No sources detected above {threshold}σ")
            return  # Exit early if no detections

        # Generate catalog and select relevant properties
        catalog = SourceCatalog(snr_map, segm_deblend)
        table = catalog.to_table()

        available_columns = set(table.colnames)
        columns_to_keep = [
            'id',
            'xcentroid', 'ycentroid',
            'area',
            'max_value',
            'sum',
        ]

        # Keep only the columns that actually exist
        columns_to_keep = [col for col in columns_to_keep if col in available_columns]
        table = table[columns_to_keep]

        # Get radial distance map
        rmap = cube.disk_coords(inc=self.inc, PA=self.PA)[0]

        # Compute radial distances for each source
        radius_list = []
        for i in range(len(catalog)):
        
            x_pix = int(round(catalog.xcentroid[i]))
            y_pix = int(round(catalog.ycentroid[i]))
            radius_arcsec = rmap[y_pix, x_pix]
            radius_au = radius_arcsec * self.distance_pc
            radius_list.append(radius_au)

        # Add as a new column
        table['radius_au'] = radius_list

        # Save to text file in organized folder structure
        output_base = "Disk_Residual_Profile"
        disk_output_dir = os.path.join(output_base, self.name, f"robust_{robust_val}")
        os.makedirs(disk_output_dir, exist_ok=True)

        filename = os.path.join(
            disk_output_dir, 
            f"source_catalog_{self.name}_robust{robust_val}_thresh{threshold_str}.txt"
        )
        ascii.write(table, filename, format='commented_header', overwrite=True)
        print(f"  Saved source catalog to {filename}")

