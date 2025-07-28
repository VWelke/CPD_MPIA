
    




#-------------------------------------------------------
#  Plot the SNR maps and contour the high SNR regions
#-------------------------------------------------------

def plot_snr_map_single(disk_name, snr_map, robust_val, threshold=5.0, save_plots=True, output_dir="SNR_plots"):
    """
    Plot SNR map and detected sources for a SINGLE disk.
    
    Parameters:
    - disk_name: Name of the disk
    - snr_map: 2D SNR array for this disk
    - robust_val: Robust value used
    - threshold: Detection threshold for visualization
    - save_plots: Whether to save plots
    - output_dir: Directory to save plots
    """

    # Get the size of the pixel to estimate CPD size in pixels
    cube = disk_obj.get_cube(robust_val, cube_type="residual")
    pixel_scale_deg = abs(cube.header['CDELT1'])
    pixel_scale_arcsec = pixel_scale_deg * 3600
    print(f"  Pixel scale: {pixel_scale_arcsec:.3f} arcsec/pixel")
    
    # Convert to AU if distance available
    if hasattr(disk_obj, 'distance_pc') and disk_obj.distance_pc:
        pixel_scale_au = pixel_scale_arcsec * disk_obj.distance_pc
        print(f"  Pixel scale: {pixel_scale_au:.1f} AU/pixel")

    
    # Use photutils to detect sources
    segm = detect_sources(snr_map, threshold, npixels=2, connectivity=8)  
    
    if segm is not None:
        segm_deblend = deblend_sources(snr_map, segm, npixels=5, connectivity=8)
        print(f"  Detected {segm_deblend.nlabels} sources")
    else:
        segm_deblend = None
        print(f"  No sources detected above {threshold}σ")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{disk_name} - SNR Analysis (robust={robust_val})', fontsize=16)
    
    # Original SNR map
    im1 = axes[0,0].imshow(snr_map, origin='lower', cmap='RdBu_r', vmin=-5, vmax=5)
    axes[0,0].set_title('SNR Map')
    axes[0,0].set_xlabel('Pixel X')
    axes[0,0].set_ylabel('Pixel Y')
    plt.colorbar(im1, ax=axes[0,0], label='SNR')
    
    # Segmentation map
    if segm_deblend is not None:
        im2 = axes[0,1].imshow(segm_deblend, origin='lower', cmap='rainbow')
        axes[0,1].set_title(f'Detected Sources ({segm_deblend.nlabels} total)')
    else:
        axes[0,1].imshow(np.zeros_like(snr_map), origin='lower', cmap='gray')
        axes[0,1].set_title('No Sources Detected')
    axes[0,1].set_xlabel('Pixel X')
    axes[0,1].set_ylabel('Pixel Y')
    plt.colorbar(im2 if segm_deblend is not None else axes[0,1].images[0], 
                ax=axes[0,1], label='Source ID')
    
    # SNR map with contours
    axes[1,0].imshow(snr_map, origin='lower', cmap='RdBu_r', vmin=-5, vmax=5)
    if segm_deblend is not None:
        axes[1,0].contour(segm_deblend, levels=np.arange(1, segm_deblend.nlabels+1), 
                         colors='white', linewidths=1, alpha=0.8)
    axes[1,0].set_title('SNR + Source Contours')
    axes[1,0].set_xlabel('Pixel X')
    axes[1,0].set_ylabel('Pixel Y')
    
    # High SNR regions
    high_snr = np.where(snr_map > threshold, snr_map, 0)
    vmax = np.max(high_snr) if np.max(high_snr) > 0 else 1  # Avoid vmax=0
    im4 = axes[1,1].imshow(high_snr, origin='lower', cmap='plasma', vmin=0, vmax=vmax)
    axes[1,1].set_title(f'High SNR Regions (>{threshold}σ)')
    axes[1,1].set_xlabel('Pixel X')
    axes[1,1].set_ylabel('Pixel Y')
    plt.colorbar(im4, ax=axes[1,1], label='SNR')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'{output_dir}/{disk_name}_SNR_analysis_robust{robust_val}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()







#-----------------------------------------------
# Catalog the clumps into a table and txt
#-----------------------------------------------




#------------------------------------------------
# Plot distribution of clump properties
#------------------------------------------------

print(f" SNR range: {np.nanmin(snr_map):.2f} to {np.nanmax(snr_map):.2f}")
print(f" Pixels > {scale_factor}σ: {np.sum(snr_map > scale_factor)}")


# Analyze radial distribution of sources
source_radii = []
source_bins = []

for i in range(len(cat)):
    x_pix = int(cat.xcentroid[i])
    y_pix = int(cat.ycentroid[i])
    radius_arcsec = rmap[y_pix, x_pix]
    radius_au = radius_arcsec * radius_factor
    bin_index = np.digitize(radius_arcsec, rbins) - 1
    bin_index = max(0, min(bin_index, len(dy)-1))
    
    source_radii.append(radius_au)
    source_bins.append(bin_index)

# Plot histogram of source detections vs radius
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram of sources by radius
ax.hist(source_radii, bins=20, alpha=0.7, edgecolor='black')
ax.set_xlabel('Radius (AU)')
ax.set_ylabel('Number of Sources')
ax.set_title(f'{disk_name}: Distribution of High SNR Sources vs Radius')

plt.tight_layout()
plt.show()

print(f"\nSummary:")
print(f"Sources found at radii: {min(source_radii):.1f} - {max(source_radii):.1f} AU")
print(f"Most sources in radial bin: {max(set(source_bins), key=source_bins.count)}")





import matplotlib.pyplot as plt
import numpy as np
import os
from photutils import detect_sources, deblend_sources

def plot_snr_map_simple(disk_name, disk_obj, snr_map, robust_val, threshold3=3.0, threshold5=5.0, save_plots=True, output_dir="SNR_plots"):
    """
    Plot simplified SNR map with 3σ and 5σ contours.

    Parameters:
    - disk_name: Name of the disk
    - disk_obj: The disk object to access header info
    - snr_map: 2D SNR array
    - robust_val: Robust parameter as string
    - threshold3: 3σ contour level
    - threshold5: 5σ contour level
    - save_plots: Whether to save the plot
    - output_dir: Directory to save output
    """

    # Get pixel scale from header
    cube = disk_obj.get_cube(robust_val, cube_type="residual")
    pixel_scale_arcsec = abs(cube.header['CDELT1']) * 3600
    print(f"  Pixel scale: {pixel_scale_arcsec:.3f} arcsec/pixel")

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create binary masks for 3σ and 5σ
    mask_3sigma = snr_map >= threshold3
    mask_5sigma = snr_map >= threshold5

    # Plot
    plt.figure(figsize=(8, 7))
    plt.imshow(snr_map, origin='lower', cmap='gray', vmin=-5, vmax=5)
    plt.colorbar(label='SNR')
    
    # Contours
    plt.contour(mask_3sigma, levels=[0.5], colors='blue', linewidths=1.5, linestyles='--', label='>3σ')
    plt.contour(mask_5sigma, levels=[0.5], colors='red', linewidths=1.5, linestyles='-', label='>5σ')

    plt.title(f"{disk_name} — SNR Contours (robust={robust_val})")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.grid(False)

    # Legend (manual)
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='>3σ'),
        Line2D([0], [0], color='red', lw=2, linestyle='-', label='>5σ')
    ]
    plt.legend(handles=custom_lines, loc='upper right')

    if save_plots:
        fname = os.path.join(output_dir, f"{disk_name}_SNR_contours_robust{robust_val}.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()