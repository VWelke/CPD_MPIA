import os
import numpy as np
from casatools import table
import shutil

tb = table()

def ImportMS(msfile, modelfile, suffix='model', make_resid=False):
    """
    Copy a measurement set and replace its DATA column with model visibilities.
    Optionally create a residual MS (DATA – model).
    """
    filename = msfile
    if not filename.endswith('.ms'):
        print("MS name must end in '.ms'")
        return

    MS_filename = filename[:-3]
    out_ms = f"{MS_filename}.{suffix}.ms"

    # Remove and copy
    if os.path.exists(out_ms):
        shutil.rmtree(out_ms)
    shutil.copytree(filename, out_ms)

    # Read data
    tb.open(out_ms)
    data = tb.getcol("DATA")
    flag = tb.getcol("FLAG")
    tb.close()

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

    # Inject model visibilities
    data[:, :, unflagged] = mdl

    # --- Correct writing with putvarcol() ---
    tb.open(out_ms, nomodify=False)
    nrows = tb.nrows()
    vardata = {r: data[:, :, r] for r in range(nrows)}
    tb.putvarcol("DATA", vardata)
    tb.flush()
    tb.close()

    # --- Residual creation (optional) ---
    if make_resid:
        resid_ms = f"{MS_filename}.resid.ms"
        if os.path.exists(resid_ms):
            shutil.rmtree(resid_ms)
        shutil.copytree(filename, resid_ms)

        tb.open(resid_ms)
        data = tb.getcol("DATA")
        flag = tb.getcol("FLAG")
        tb.close()

        unflagged = np.squeeze(np.any(flag, axis=0) == False)
        data[:, :, unflagged] -= mdl

        tb.open(resid_ms, nomodify=False)  # ✅ fixed: write to residual MS
        nrows = tb.nrows()
        vardata = {r: data[:, :, r] for r in range(nrows)}
        tb.putvarcol("DATA", vardata)
        tb.flush()
        tb.close()