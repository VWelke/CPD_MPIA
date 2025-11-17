import os
import matplotlib.pyplot as plt

import numpy as np
from math import sqrt
from casatools import table
tb= table()
from casatasks import split, flagdata, flagmanager, tclean, imfit, imhead, imstat, gencal, applycal


def export_MS(msfile):
    """
    Spectrally averages visibilities to a single channel per SPW and exports to .npz file

    msfile: Name of CASA measurement set, ending in '.ms' (string)
    """
    filename = msfile
    if filename[-3:]!='.ms':
        print("MS name must end in '.ms'")
        return
    # strip off the '.ms'
    MS_filename = filename.replace('.ms', '')

    # get information about spectral windows

    tb.open(MS_filename+'.ms/SPECTRAL_WINDOW')
    num_chan = tb.getcol('NUM_CHAN').tolist()
    tb.close()

    # spectral averaging (1 channel per SPW)
    # num_chan is a list of number of channels per SPW eg [1,1,8,1,1,8,8,1,1,1]
    # and width = num_chan means that the width of each SPW is the full number of channels in that SPW, and so the output MS will have 1 channel per SPW
    # if I let width = 1, then the output MS will have the same number of channels as the input MS
    os.system('rm -rf %s' % MS_filename+'_spavg.ms')
    split(vis=MS_filename+'.ms', width=1, datacolumn='data',
    outputvis=MS_filename+'_spavg.ms')

    # get the data tables
    tb.open(MS_filename+'_spavg.ms')
    data   = np.squeeze(tb.getcol("DATA"))
    flag   = np.squeeze(tb.getcol("FLAG"))
    uvw    = tb.getcol("UVW")
    weight = tb.getcol("WEIGHT")
    spwid  = tb.getcol("DATA_DESC_ID")
    tb.close()

    # get frequency information
    tb.open(MS_filename+'_spavg.ms/SPECTRAL_WINDOW')
    freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
    tb.close()

    # get rid of any flagged columns
    good   = np.squeeze(np.any(flag, axis=0)==False)
    data   = data[:,good]
    weight = weight[:,good]
    uvw    = uvw[:,good]
    spwid = spwid[good]
    
    # compute spatial frequencies in lambda units
    get_freq = lambda ispw: freqlist[ispw]
    freqs = get_freq(spwid) #get spectral frequency corresponding to each datapoint
    u = uvw[0,:] * freqs / 2.9979e8
    v = uvw[1,:] * freqs / 2.9979e8

    #average the polarizations
    Re  = np.sum(data.real*weight, axis=0) / np.sum(weight, axis=0)
    Im  = np.sum(data.imag*weight, axis=0) / np.sum(weight, axis=0)
    Vis = Re + 1j*Im
    Wgt = np.sum(weight, axis=0)

    #output to npz file and delete intermediate measurement set
    os.system('rm -rf %s' % MS_filename+'_spavg.ms')
    os.system('rm -rf '+MS_filename+'.vis.npz')
    np.savez(MS_filename+'.vis', u=u, v=v, Vis=Vis, Wgt=Wgt)
    print(("#Measurement set exported to %s" % (MS_filename+'.vis.npz',)))
