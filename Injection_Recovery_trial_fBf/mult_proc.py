from multiprocessing import Pool
from frank.fit import FrankFitter
import numpy as np

# list of your datasets
sources = [
    "AA_Tau_time_ave_continuum.vis.npz",
    "CQ_Tau_time_ave_continuum.vis.npz",
    "HD_14300_time_ave_continuum.vis.npz",
    "DL_Tau_time_ave_continuum.vis.npz"
]

def run_frank_fit(vis_file):
    """Run one frank fit."""
    print(f"Running frank on {vis_file}")
    fit = FrankFitter(vis_file)
    fit.fit()
    fit.save_fit()     # or whatever save method you use
    return vis_file

if __name__ == "__main__":
    # limit OMP threads per frank instance to 1 (important!)
    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    # run 4 fits in parallel (adjust to CPU count)
    with Pool(processes=4) as pool:
        results = pool.map(run_frank_fit, sources)

    print("All fits completed:", results)