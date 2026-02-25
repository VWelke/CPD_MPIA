(frank_env) [vawelke@Node-01 J1615_gap0]$ tail -f prepimaging.log
nohup: ignoring input
/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/lib/py/lib/python3.10/site-packages/IPython/core/interactiveshell.py:937: UserWarning: Attempting to work in a virtualenv. If you encounter problems, please install IPython inside the virtualenv.
  warn(

Using user configuration file /nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py

IPython 8.26.0 -- An enhanced Interactive Python.

No event loop hook running.
measures_update ... acquiring the lock ...
  ... finding available measures at www.astron.nl ...
  ... downloading WSRT_Measures_20260217-160001.ztar from ASTRON server to /nexus/posix0/MIA-astro-env/myben/vawelke/casa_data ...
  ... measures data updated at /nexus/posix0/MIA-astro-env/myben/vawelke/casa_data
2026-02-18 14:27:57 INFO: Environment is not MPI enabled. Pipeline operating in single host mode
2026-02-18 14:28:04 INFO: Pipeline version 2025.1.0.35 running on Node-01
2026-02-18 14:28:04 INFO: Host environment:
        CPU: AMD EPYC 9454 48-Core Processor (physical cores: 96, logical cores: 192)
        Memory: 1416.8 GiB RAM, unknown swap
        OS: AlmaLinux 9.7 (Moss Jungle Cat)
        cgroup limits: N/A of 192 CPU cores, memory limits=N/A
        ulimit limits: CPU time=N/A, memory=N/A, files=1024
2026-02-18 14:28:04 INFO: Environment as detected by CASA:
        CPUs reported by CASA: 192 cores, max 8 OpenMP threads
        Available memory: 1416.8 GiB
2026-02-18 14:28:05 INFO: Initializing cli...
2026-02-18 14:28:05 INFO: Loaded Pipeline commands from package: h
2026-02-18 14:28:05 INFO: Loaded Pipeline commands from package: hif
2026-02-18 14:28:05 INFO: Loaded Pipeline commands from package: hifa
2026-02-18 14:28:05 INFO: Loaded Pipeline commands from package: hifv
2026-02-18 14:28:05 INFO: Loaded Pipeline commands from package: hsd
2026-02-18 14:28:05 INFO: Loaded Pipeline commands from package: hsdn
CASA 6.6.6.17 -- Common Astronomy Software Applications [6.6.6.17]

CASA <1>:
CASA <2>:
CASA <3>:
CASA <3>:
CASA <4>:
CASA <5>:
CASA <5>:
CASA <6>:
CASA <7>:
CASA <7>:
CASA <8>:
CASA <8>:
CASA <9>:
CASA <10>:
CASA <10>:
CASA <11>:
CASA <12>:
CASA <13>:
CASA <13>:
CASA <13>:
CASA <14>:
CASA <14>:
CASA <15>:       ...:
CASA <16>:
CASA <17>: Out[17]: 0

CASA <18>:
CASA <18>:
CASA <19>:
CASA <20>:
CASA <21>:
CASA <21>:
CASA <21>:
CASA <22>:
CASA <23>:
CASA <24>:
CASA <24>:
CASA <25>:
CASA <26>:       ...:       ...:
CASA <27>:
CASA <27>:
CASA <28>:
CASA <29>:       ...:       ...:       ...:       ...:       ...:       ...:
0%....10....20....30....40....50....60....70....80....90....100%
2026-02-18 14:28:44     WARN    task_tclean::SynthesisImagerVi2::makePB (file /source/casa6/casatools/src/code/synthesis/ImagerObjects/SynthesisImagerVi2.cc, line 3447)        The MS has multiple antenna diameters ..PB could be wrong

0%....10....20....30....40....50....60....70....80....90....100%

0%....10....20....30....40....50....60....70....80....90....100%

0%....10....20....30....40....50....60....70....80....90....100%
Out[29]:
{'cleanstate': 'running',
 'cyclefactor': 1.0,
 'cycleiterdone': 0,
 'cycleniter': 300,
 'cyclethreshold': 1.928342680912465e-05,
 'interactiveiterdone': 52,
 'interactivemode': False,
 'interactiveniter': 0,
 'interactivethreshold': 0.0,
 'iterdone': 52,
 'loopgain': 0.30000001192092896,
 'maxpsffraction': 0.800000011920929,
 'maxpsfsidelobe': 0.2184024602174759,
 'minpsffraction': 0.05000000074505806,
 'niter': 50000,
 'nmajordone': 3,
 'nsigma': 0.0,
 'stopcode': 2,
 'summarymajor': array([ 0, 43, 52]),
 'summaryminor': {0: {0: {0: {'iterDone': [np.float64(43.0), np.float64(9.0)],
     'peakRes': [np.float64(9.610432607587427e-05),
      np.float64(8.829308353597298e-05)],
     'modelFlux': [np.float64(0.00022387578792404383),
      np.float64(0.00017239636508747935)],
     'cycleThresh': [np.float64(0.00010136725177289918),
      np.float64(8.900000102585182e-05)]}}}},
 'threshold': 8.900000102585182e-05,
 'stopDescription': 'threshold'}

CASA <30>:
CASA <31>:
CASA <31>:
CASA <31>:
CASA <32>: /nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/JvM_correction_brief.py:82: RuntimeWarning: divide by zero encountered in scalar divide
  epsilon = np.sum(clean_beam)/np.sum(psf_windowed)
2026-02-18 14:29:23     SEVERE          Exception Reported: ImageExprParse: 'inf' is an unknown lattice, image, or region
2026-02-18 14:29:23     SEVERE  +       Scanned so far: "J1615_gap0.F35uJy_0000.resid_convolved_model_temp.image"+inf*
2026-02-18 14:29:23     SEVERE  immath::::casa  Unable to process expression IM0+inf*IM1
2026-02-18 14:29:23     SEVERE  immath::::casa  Exception caught was: ImageExprParse: 'inf' is an unknown lattice, image, or region
2026-02-18 14:29:23     SEVERE  immath::::casa+ Scanned so far: "J1615_gap0.F35uJy_0000.resid_convolved_model_temp.image"+inf*
2026-02-18 14:29:23     SEVERE  immath::::casa  Task immath raised an exception of class RuntimeError with the following message: ImageExprParse: 'inf' is an unknown lattice, image, or region
2026-02-18 14:29:23     SEVERE  immath::::casa+ Scanned so far: "J1615_gap0.F35uJy_0000.resid_convolved_model_temp.image"+inf*
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[32], line 1
----> 1 eps = do_JvM_correction_and_get_epsilon(im_outfile)

File /nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/JvM_correction_brief.py:96, in do_JvM_correction_and_get_epsilon(root)
     94 except:
     95     pass
---> 96 immath(imagename=[convolved_temp_image, residual_file],
     97        expr='IM0 + ' + str(epsilon) + '*IM1', outfile=root+".JvMcorr.image")
     99 # clean up
    100 shutil.rmtree(convolved_temp_image)

File /nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/lib/py/lib/python3.10/site-packages/casatasks/immath.py:308, in _immath.__call__(self, imagename, mode, outfile, expr, varnames, sigma, polithresh, mask, region, box, chans, stokes, stretch, imagemd, prec)
    306 task_result = None
    307 try:
--> 308     task_result = _immath_t( _pc.document['imagename'], _pc.document['mode'], _pc.document['outfile'], _pc.document['expr'], _pc.document['varnames'], _pc.document['sigma'], _pc.document['polithresh'], _pc.document['mask'], _pc.document['region'], _pc.document['box'], _pc.document['chans'], _pc.document['stokes'], _pc.document['stretch'], _pc.document['imagemd'], _pc.document['prec'] )
    309 except Exception as exc:
    310     _except_log('immath', exc)

File /nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/lib/py/lib/python3.10/site-packages/casatasks/private/task_immath.py:248, in immath(imagename, mode, outfile, expr, varnames, sigma, polithresh, mask, region, box, chans, stokes, stretch, imagemd, prec)
    242         outia = _immath_compute(
    243             imagename, expr, outfile, imagemd, myia, prec
    244         )
    245     else:
    246         # If the user didn't give any region or mask information
    247         # then just evaluated the expression with the filenames in it.
--> 248         outia = _immath_dofull(
    249             imagename, imagemd, outfile, mode, expr,
    250             varnames, filenames, myia, prec
    251         )
    252 else:
    253     raise(Exception, "Unsupported mode " + str(mode))

File /nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/lib/py/lib/python3.10/site-packages/casatasks/private/task_immath.py:488, in _immath_dofull(imagename, imagemd, outfile, mode, expr, varnames, filenames, myia, prec)
    483 def _immath_dofull(
    484     imagename, imagemd, outfile, mode, expr,
    485     varnames, filenames, myia, prec
    486 ):
    487     expr = _immath_expr_from_varnames(expr, varnames, filenames)
--> 488     return _immath_compute(
    489         imagename, expr, outfile, imagemd, myia, prec
    490     )

File /nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/lib/py/lib/python3.10/site-packages/casatasks/private/task_immath.py:421, in _immath_compute(imagename, expr, outfile, imagemd, myia, prec)
    417 def _immath_compute(
    418     imagename, expr, outfile, imagemd, myia, prec
    419 ):
    420     # Do the calculation
--> 421     res = myia.imagecalc(
    422         pixels=expr, outfile=outfile,
    423         imagemd=_immath_translate_imagemd(imagename, imagemd), prec=prec
    424     )
    425     res.dohistory(False)
    426     # modify stokes type for polarization intensity image

File /nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/lib/py/lib/python3.10/site-packages/casatools/image.py:108, in image.imagecalc(self, outfile, pixels, overwrite, imagemd, prec)
     46 def imagecalc(self, outfile='', pixels='', overwrite=False, imagemd='', prec='float'):
     47     """This method is used to evaluate a mathematical expression involving
     48     existing images. It fully supports float, double, and complex float, and complex
     49     double valued images.
   (...)
    106
    107     """
--> 108     return _wrap_image(swig_object=self._swigobj.imagecalc(outfile, pixels, overwrite, imagemd, prec))

File /nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/lib/py/lib/python3.10/site-packages/casatools/__casac__/image.py:354, in image.imagecalc(self, *args, **kwargs)
    194 def imagecalc(self, *args, **kwargs):
    195     """
    196     imagecalc(self, _outfile, _pixels, _overwrite, _imagemd, _prec) -> image
    197
   (...)
    352
    353     """
--> 354     return _image.image_imagecalc(self, *args, **kwargs)

RuntimeError: ImageExprParse: 'inf' is an unknown lattice, image, or region
Scanned so far: "J1615_gap0.F35uJy_0000.resid_convolved_model_temp.image"+inf*

CASA <33>:
CASA <33>:
CASA <34>:       ...:
CASA <35>:
CASA <35>:
CASA <36>:       ...:       ...: Out[36]: 0

CASA <37>: Out[37]: 0

CASA <38>: Out[38]: 0

CASA <39>: Out[39]: 0

CASA <40>: Out[40]: 0

CASA <41>:
CASA <41>: 74.78048014640808

CASA <42>: Do you really want to exit ([y]/n)?
/nexus/posix0/MIA-astro-env/myben/vawelke/software/casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8/lib/py/lib/python3.10/site-packages/IPython/core/interactiveshell.py:937: UserWarning: Attempting to work in a virtualenv. If you encounter problems, please install IPython inside the virtualenv.
  warn(

Using user configuration file /nexus/posix0/MIA-astro-env/myben/vawelke/casa_config.py

IPython 8.26.0 -- An enhanced Interactive Python.

No event loop hook running.
2026-02-18 14:29:36 INFO: Environment is not MPI enabled. Pipeline operating in single host mode
2026-02-18 14:29:39 INFO: Pipeline version 2025.1.0.35 running on Node-01
2026-02-18 14:29:39 INFO: Host environment:
        CPU: AMD EPYC 9454 48-Core Processor (physical cores: 96, logical cores: 192)
        Memory: 1416.8 GiB RAM, unknown swap
        OS: AlmaLinux 9.7 (Moss Jungle Cat)
        cgroup limits: N/A of 192 CPU cores, memory limits=N/A
        ulimit limits: CPU time=N/A, memory=N/A, files=1024
2026-02-18 14:29:39 INFO: Environment as detected by CASA:
        CPUs reported by CASA: 192 cores, max 8 OpenMP threads
        Available memory: 1416.8 GiB
2026-02-18 14:29:40 INFO: Initializing cli...
2026-02-18 14:29:40 INFO: Loaded Pipeline commands from package: h
2026-02-18 14:29:40 INFO: Loaded Pipeline commands from package: hif
2026-02-18 14:29:40 INFO: Loaded Pipeline commands from package: hifa
2026-02-18 14:29:40 INFO: Loaded Pipeline commands from package: hifv
2026-02-18 14:29:40 INFO: Loaded Pipeline commands from package: hsd
2026-02-18 14:29:40 INFO: Loaded Pipeline commands from package: hsdn
CASA 6.6.6.17 -- Common Astronomy Software Applications [6.6.6.17]

CASA <1>:
CASA <2>:
CASA <3>:
CASA <3>:
CASA <4>:
CASA <5>:
CASA <5>:
CASA <6>:
CASA <6>:      ...: 2026-02-18 14:29:41 WARN    importfits::::casa      This image has no beam or angular resolution provided, so you will not receive warnings from
2026-02-18 14:29:41     WARN    importfits::::casa+     tasks such as imregrid if your image pixels do not sample the the angular resolution well.
2026-02-18 14:29:41     WARN    importfits::::casa+     (This only affects warnings, not any functionality).
2026-02-18 14:29:41     WARN    importfits::::casa+     Providing a beam and brightness units in an image can also be useful for flux calculations.
2026-02-18 14:29:41     WARN    importfits::::casa+     If you wish to add a beam or brightness units to your image, please use
2026-02-18 14:29:41     WARN    importfits::::casa+     the "beam" parameter or ia.setrestoringbeam() and ia.setbrightnessunit()

CASA <7>: Do you really want to exit ([y]/n)?
