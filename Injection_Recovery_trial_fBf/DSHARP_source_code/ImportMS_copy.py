import os
import shutil
import numpy as np
import site, sys
sys.path.append(site.USER_BASE + "/lib/python3.8/site-packages")
from tqdm import tqdm
from casatools import table


def ImportMS(msfile, modelfile, suffix="model", make_resid=False, chunk_size=10000):
    """
    Copy a Measurement Set and replace its DATA column with model visibilities.
    Handles mixed-SPW MS (1 vs 8 channels) and avoids CASA table handle leaks.

    Parameters
    ----------
    msfile : str
        Path to the input Measurement Set (ends with .ms)
    modelfile : str
        Path to the NPZ file containing model visibilities (expects key 'V')
    suffix : str
        Suffix for the output MS (default 'model')
    make_resid : bool
        If True, also produce a residual MS (DATA - model)
    chunk_size : int
        Number of rows to write per chunk when updating DATA
    """

    if not msfile.endswith(".ms"):
        raise ValueError("Input msfile must end with '.ms'")

    MS_filename = msfile[:-3]
    out_ms = f"{MS_filename}.{suffix}.ms"

    # --- Copy MS safely ---
    if os.path.exists(out_ms):
        shutil.rmtree(out_ms)
    shutil.copytree(msfile, out_ms)

    # --- Read SPW info safely ---
    print("\n=== SPW Summary (from SPECTRAL_WINDOW table) ===")
    spw_tb = table()
    spw_tb.open(os.path.join(out_ms, "SPECTRAL_WINDOW"))
    num_chan = spw_tb.getcol("NUM_CHAN") # array of number of channels per SPW
    ref_freq = spw_tb.getcol("REF_FREQUENCY")
    spw_tb.close()
    for i, (nc, rf) in enumerate(zip(num_chan, ref_freq)):
        print(f" SPW {i:02d}: {nc:>3} chans, ref freq = {rf/1e9:.3f} GHz")
    print("===============================================\n")

    # --- Load model visibilities (NumPy only, no CASA handle open) ---
    mdl = np.load(modelfile + ".npz")["V"] 
    if mdl.ndim == 1:
        mdl = mdl[np.newaxis, np.newaxis, :]
    print(f"Loaded model shape: {mdl.shape}")

    # --- Inspect main table and handle mixed-SPW safely ---
    tb = table()
    tb.open(out_ms)
    nrows = tb.nrows()
    first_cell = tb.getcell("DATA", 0)
    npol, nchan = first_cell.shape
    print(f"First DATA cell shape: {first_cell.shape}")
    tb.close()

    # Allocate arrays for safe uniform read
    data = np.zeros((npol, 1, nrows), dtype=np.complex128)
    flag = np.ones((npol, 1, nrows), dtype=bool)
    valid_rows = []

    tb = table()
    tb.open(out_ms)
    for r in range(nrows):
        try:
            dcell = tb.getcell("DATA", r)
            if dcell.shape[1] == mdl.shape[-1] or dcell.shape[1] == 1:
                data[:, :, r] = dcell[:, :1]
                flag[:, :, r] = False
                valid_rows.append(r)
        except Exception as e:
            print(f"Skipping row {r} due to shape error: {e}")
    tb.close()

    n_valid = len(valid_rows)
    print(f"✅ Loaded {n_valid} valid continuum rows (of {nrows} total).")
    print(f"Replacing with model shape {mdl.shape}.\n")

    # --- Broadcast model to match DATA shape ---
    if mdl.shape[2] != 1:
        mdl = mdl[:, :, :1]  # ensure same shape for writing

    # --- Write updated DATA in chunks ---
    tb = table()
    tb.open(out_ms, nomodify=False)
    for start in tqdm(range(0, n_valid, chunk_size), desc="Writing DATA"):
        rchunk = valid_rows[start:start + chunk_size]
        chunk_data = np.ascontiguousarray(data[:, :, rchunk])
        for i, row_idx in enumerate(rchunk):
            tb.putcell("DATA", row_idx, chunk_data[:, :, i])
    tb.flush()
    tb.close()
    print("✅ Finished writing model visibilities.\n")

    # --- Optional residual MS ---
    if make_resid:
        resid_ms = f"{MS_filename}.resid.ms"
        if os.path.exists(resid_ms):
            shutil.rmtree(resid_ms)
        shutil.copytree(msfile, resid_ms)

        tb = table()
        tb.open(resid_ms)
        nrows = tb.nrows()
        resid_data = np.zeros((npol, 1, nrows), dtype=np.complex128)
        for r in valid_rows:
            resid_data[:, :, r] = tb.getcell("DATA", r)[:, :1] - mdl[:, :, 0:1]
        tb.close()

        tb = table()
        tb.open(resid_ms, nomodify=False)
        for start in tqdm(range(0, n_valid, chunk_size), desc="Writing Residuals"):
            rchunk = valid_rows[start:start + chunk_size]
            chunk_data = np.ascontiguousarray(resid_data[:, :, rchunk])
            for i, row_idx in enumerate(rchunk):
                tb.putcell("DATA", row_idx, chunk_data[:, :, i])
        tb.flush()
        tb.close()
        print("✅ Finished writing residual visibilities.\n")

    print("ImportMS completed successfully.")
