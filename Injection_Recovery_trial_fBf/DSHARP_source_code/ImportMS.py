import os
import numpy as np
import shutil
from casatools import table

tb = table()

def ImportMS(msfile, modelfile, suffix='model', make_resid=False):
    """
    Copy a measurement set and replace its DATA column with model visibilities.
    Optionally create a residual MS (DATA – model).
    Supports mixed-SPW MS files (different channel counts per SPW).
    """

    if not msfile.endswith('.ms'):
        print("MS name must end in '.ms'")
        return

    MS_filename = msfile[:-3]
    out_ms = f"{MS_filename}.{suffix}.ms"

    # --- Safe copy of the MS ---
    if os.path.exists(out_ms):
        shutil.rmtree(out_ms)
    shutil.copytree(msfile, out_ms)

    # --- Read DATA and FLAG from copy ---
    tb.open(out_ms)
    data = tb.getcol("DATA")
    flag = tb.getcol("FLAG")
    spw_ids = tb.getcol("DATA_DESC_ID")
    nrows = tb.nrows()
    tb.close()

    # --- Read SPW structure to get channel counts ---
    tb2 = table()
    tb2.open(out_ms + "/SPECTRAL_WINDOW")
    nchan_per_spw = tb2.getcol("NUM_CHAN")
    tb2.close()
    print(f"Unique nchan in MS: {np.unique(nchan_per_spw)}")

    # --- Load model ---
    mdl = np.load(modelfile + '.npz')['V']
    if mdl.ndim == 1:
        mdl = mdl[np.newaxis, np.newaxis, :]  # (1,1,N)
        mdl = np.broadcast_to(mdl, (data.shape[0], 1, mdl.shape[-1]))

    print("MS DATA shape:", data.shape)
    print("FLAG shape:", flag.shape)
    print("Model visibilities shape:", mdl.shape)

    # --- Determine unflagged rows ---
    unflagged = np.squeeze(np.any(flag, axis=0) == False)
    rows_to_write = np.where(unflagged)[0]
    print(f"Writing {len(rows_to_write)} unflagged rows out of {nrows} total...")

    # --- Inject model into MS per SPW ---
    tb.open(out_ms, nomodify=False)

    for i, r in enumerate(rows_to_write):
        spw = spw_ids[r]
        nchan = nchan_per_spw[spw]

        # take model for this row, expand channels if needed
        arr = mdl[:, 0, r:r+1]  # (2,1)
        if nchan > 1:
            arr = np.repeat(arr, nchan, axis=1)

        arr = np.ascontiguousarray(arr).astype(np.complex128, copy=False)
        tb.putcell("DATA", int(r), arr)

        if (i + 1) % 100000 == 0:
            print(f"  Written {i+1}/{len(rows_to_write)} rows...")

    tb.flush()
    tb.close()
    print("✅ Finished writing model visibilities.\n")

    # --- Optional residual creation ---
    if make_resid:
        resid_ms = f"{MS_filename}.resid.ms"
        if os.path.exists(resid_ms):
            shutil.rmtree(resid_ms)
        shutil.copytree(msfile, resid_ms)

        tb.open(resid_ms)
        data = tb.getcol("DATA")
        flag = tb.getcol("FLAG")
        spw_ids = tb.getcol("DATA_DESC_ID")
        nrows = tb.nrows()
        tb.close()

        tb2.open(resid_ms + "/SPECTRAL_WINDOW")
        nchan_per_spw = tb2.getcol("NUM_CHAN")
        tb2.close()

        unflagged = np.squeeze(np.any(flag, axis=0) == False)
        rows_to_write = np.where(unflagged)[0]

        tb.open(resid_ms, nomodify=False)
        for i, r in enumerate(rows_to_write):
            spw = spw_ids[r]
            nchan = nchan_per_spw[spw]

            arr = mdl[:, 0, r:r+1]
            if nchan > 1:
                arr = np.repeat(arr, nchan, axis=1)

            arr = np.ascontiguousarray(arr).astype(np.complex128, copy=False)
            orig = tb.getcell("DATA", int(r))
            tb.putcell("DATA", int(r), orig - arr)

            if (i + 1) % 100000 == 0:
                print(f"  Written {i+1}/{len(rows_to_write)} residual rows...")

        tb.flush()
        tb.close()
        print("✅ Finished writing residual visibilities.\n")