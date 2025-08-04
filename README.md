# CPD_MPIA
This project analyzes circumplanetary disk (CPD) signals by processing residual data, modeling emission limits, and exploring parameter space. It aims to assess CPD detectability in observational data and may include testing or extending existing CPD emission models.

-> will re-allocate the functions after the the entire class is defined 

CPD_MPIA/
# this is the inital plan 
├─ ALMA_Disk_Python_Toolkit/
|   ├─ __init__.py
|   ├─ disk.py          # thin class + shared state; delegates to other modules
|   ├─ io_cubes.py      # geometry/radii/ringgap loaders + residual/clean cube access
|   ├─ analysis.py      # sigma masks, SNR maps, profiles plotting 
|   └─ detection.py     # source detection 
├─ Median_SNR/
│  ├─ disk_residuals_median_SNR.py      # will be mined/split into exalma/*
│  ├─ SNR_FITS_Maps/                     # outputs
│  └─ Disk_Residual_Profile_Median_SNR/  # outputs
├─ Standard_dy_SNR/
│  └─ Disk_Residual_Profile/             # outputs
├─ notebooks/
│  └─ Residual_Plot.ipynb