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


class DiskResiduals:
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
        Load the ringgap profile from a text file.
        The file should contain radius and intensity columns.
        """
        arr = np.loadtxt(ringgap_path, comments="#")
        self.ringgap_profile = {
            "radius_au": arr[:, 0],  # column 0 is radius in au
            "intensity_Jy_sr": arr[:, 1],  # column 1 is intensity in Jy/sr
            "d_intensity_Jy_sr": arr[:, 2]  # column 2 is uncertainty in Jy/sr
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

    def get_cube(self, robust_val, FOV=10.0, cube_type="residual"):
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
            if os.path.exists(full_path2):
                return imagecube(full_path2, FOV=FOV)
            elif os.path.exists(full_path1):
                return imagecube(full_path1, FOV=FOV)
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

        
        # Get profiles (x in arcsec)
        x_cl, y_cl, dy_cl = cube_clean.radial_profile(
            inc=self.inc, PA=self.PA, unit='Jy/beam',assume_correlated=True
        )
        x_res, y_res, dy_res = cube_resid.radial_profile(
            inc=self.inc, PA=self.PA, unit='Jy/beam',assume_correlated=True
        )

            # Convert radius unit
        if radius_unit == "au":
            x_cl = x_cl * self.distance_pc
            x_res = x_res * self.distance_pc
            xlabel = "Radius (au)"
        # Convert gaps/rings to au
            gap_unit_factor = self.distance_pc
        else:
            xlabel = "Radius (arcsec)"
            gap_unit_factor = 1.0

        # Plot
        fig, ax = plt.subplots(constrained_layout=True, figsize = (12, 6))
        ax.plot(x_cl, y_cl, color='gray', linewidth=2, label='CLEAN')
        ax.errorbar(x_cl, y_cl, dy_cl, fmt='none', ecolor='gray', alpha=0.4, capsize=2)
        ax.plot(x_res, y_res, color='crimson', linewidth=2, label='Residual')
        ax.errorbar(x_res, y_res, dy_res, fmt='none', ecolor='crimson', alpha=0.4 ,capsize=2)

         # Overlay R90 vertical line and error band
        if hasattr(self, "disksize"):
            R90 = self.disksize["R90"] * gap_unit_factor
            err_low = self.disksize["R90_err_low"] * gap_unit_factor
            err_high = self.disksize["R90_err_high"] * gap_unit_factor
            ax.axvline(R90, color='k', linestyle='--', label='R90')
            ax.axvspan(R90 - err_low, R90 + err_high, color='k', alpha=0.15, label='R90 error')

        # Overlay gaps/rings vertical line and width band
        if hasattr(self, "ringgap") and self.ringgap is not None and self.ringgap.size > 0:
            ringgap_arr = self.ringgap
            if ringgap_arr.ndim == 1:
                ringgap_arr = ringgap_arr[np.newaxis, :]
            gap_label_added = False
            ring_label_added = False
            for row in ringgap_arr:
                rad = row[1] * gap_unit_factor
                flag = int(row[2])
                width = row[4] * gap_unit_factor if not np.isnan(row[4]) else None
                                
                color = '#b9fbc0' if flag == 0 else '#cdb4fe'  # sage green for gaps, lavender for rings
                
                if flag == 0:
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
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        ax.set_title(f"{self.name} robust={robust_val}")
        plt.show()
        
        