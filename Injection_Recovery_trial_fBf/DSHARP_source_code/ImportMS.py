import os
import numpy as np
from casatools import table
import shutil
from tqdm import tqdm  # <-- add this for progress bar

tb = table()

def ImportMS(msfile, modelfile, suffix='model', make_resid=False, chunk_size=10000):
    """
    Copy a measurement set and replace its DATA column with model visibilities.
    Optionally create a residual MS (DATA – model).
    Uses fast chunked writing with tqdm progress bar.
    """
    if not msfile.endswith('.ms'):
        print("MS name must end in '.ms'")
        return

    MS_filename = msfile[:-3]
    out_ms = f"{MS_filename}.{suffix}.ms"

    # --- Copy MS safely ---
    if os.path.exists(out_ms):
        shutil.rmtree(out_ms)
    shutil.copytree(msfile, out_ms)

    # --- Read DATA + FLAG ---
    tb.open(out_ms)
    data = tb.getcol("DATA")
    flag = tb.getcol("FLAG")
    tb.close()

    # --- Prepare model ---
    unflagged = np.squeeze(np.any(flag, axis=0) == False)
    mdl = np.load(modelfile + '.npz')['V']

    if mdl.ndim == 1:
        mdl = mdl[np.newaxis, np.newaxis, :]
        mdl = np.broadcast_to(mdl, (data.shape[0], data.shape[1], mdl.shape[-1]))

    print("MS DATA shape:", data.shape)
    print("FLAG shape:", flag.shape)
    print("Unflagged mask shape:", unflagged.shape)
    print("Model visibilities shape:", mdl.shape)
    print("Target slice shape:", data[:, :, unflagged].shape)

    # --- Inject model ---
    data[:, :, unflagged] = mdl

    # --- Chunked writing with progress bar ---
    tb.open(out_ms, nomodify=False)
    nrows = tb.nrows()
    rows_to_write = np.where(unflagged)[0]
    print(f"Writing {len(rows_to_write)} unflagged rows out of {nrows} total (chunked {chunk_size})...")

    for start in tqdm(range(0, len(rows_to_write), chunk_size), desc="Writing DATA"):
        rchunk = rows_to_write[start:start + chunk_size]
        chunk_data = np.ascontiguousarray(data[:, :, rchunk])
        tb.putcol("DATA", chunk_data, startrow=int(rchunk[0]), nrow=len(rchunk))

    tb.flush()
    tb.close()
    print("✅ Finished writing model visibilities.\n")

    # --- Residual creation (optional) ---
    if make_resid:
        resid_ms = f"{MS_filename}.resid.ms"
        if os.path.exists(resid_ms):
            shutil.rmtree(resid_ms)
        shutil.copytree(msfile, resid_ms)

        tb.open(resid_ms)
        data = tb.getcol("DATA")
        flag = tb.getcol("FLAG")
        tb.close()

        unflagged = np.squeeze(np.any(flag, axis=0) == False)
        data[:, :, unflagged] -= mdl

        tb.open(resid_ms, nomodify=False)
        rows_to_write = np.where(unflagged)[0]
        print(f"Writing {len(rows_to_write)} unflagged rows to residual MS (chunked {chunk_size})...")

        for start in tqdm(range(0, len(rows_to_write), chunk_size), desc="Writing Residuals"):
            rchunk = rows_to_write[start:start + chunk_size]
            chunk_data = np.ascontiguousarray(data[:, :, rchunk])
            tb.putcol("DATA", chunk_data, startrow=int(rchunk[0]), nrow=len(rchunk))

        tb.flush()
        tb.close()
        print("✅ Finished writing residual visibilities.\n")