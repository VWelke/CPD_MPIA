This folder experiments with median-based source detection, ensuring that results from the standard noise estimation method are not overwritten. The advantage of this approach is that asymmetric features in protoplanetary disks are less likely to cause an overestimation of the noise per radial bin due to azimuthal averaging.


1. Create a new folder: Median_SNR to run the code.

2. Copy the notebook with changes on 

3. **Edit the Copy**  
   Open `Residual_Plot_median/Residual_Plot.ipynb` and modify the relevant code to use the median and the 16th–84th percentile range for standard deviation (see below).

4. **Implement Median-Based Detection**  
   In the copied notebook, locate the code that computes the standard deviation for noise estimation (e.g., in `radial_profile` or `create_sigma_mask`).  
   Change the relevant function call to use percentiles:
   ```python
   # Example: Use percentiles=True for robust std estimation
   x, y, dy = cube.radial_profile(
       inc=all_disks[disk_name].inc,
       PA=all_disks[disk_name].PA,
       unit='Jy/beam',
       percentiles=True  # <-- Add this argument
   )