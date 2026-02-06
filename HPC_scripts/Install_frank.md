```python
[vawelke@Node-01 J1852_gap0_trial1]$ python -u J1852_gap0_injectloop.py
Traceback (most recent call last):
  File "/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/J1852_gap0_trial1/J1852_gap0_injectloop.py", line 7, in <module>
    from frank.geometry import FixedGeometry
ModuleNotFoundError: No module named 'frank'
[vawelke@Node-01 J1852_gap0_trial1]$ cd ..
[vawelke@Node-01 inj_rev]$ cd ..
[vawelke@Node-01 vawelke]$ NEXUS=/nexus/posix0/MIA-astro-env/myben/vawelke
[vawelke@Node-01 vawelke]$ mkdir -p $NEXUS/.conda/pkgs
[vawelke@Node-01 vawelke]$ mkdir -p $NEXUS/.conda/envs
[vawelke@Node-01 vawelke]$ conda config --add pkgs_dirs $NEXUS/.conda/pkgs
-bash: conda: command not found
[vawelke@Node-01 vawelke]$ [vawelke@Node-01 vawelke]$ conda config --add pkgs_dirs $NEXUS/.conda/pkgs
-bash: conda: command not found
[vawelke@Node-01 vawelke]which conda
-bash: [vawelke@Node-01: command not found
-bash: -bash:: command not found
-bash: [vawelke@Node-01: command not found
[vawelke@Node-01 vawelke]$ module avail conda
No module(s) or extension(s) found!
If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".


[vawelke@Node-01 vawelke]$ module avail miniconda
No module(s) or extension(s) found!
If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".


[vawelke@Node-01 vawelke]$ module available anaconda
No module(s) or extension(s) found!
If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".


[vawelke@Node-01 vawelke]$ ml -d av

----------------------------------------------- /usr/share/modulefiles ------------------------------------------------
   mpi/openmpi-x86_64

---------------------------------------- /usr/share/lmod/lmod/modulefiles/Core ----------------------------------------
   lmod    settarg

If the avail list is too long consider trying:

"module --default avail" or "ml -d av" to just list the default modules.
"module overview" or "ml ov" to display the number of modules for each name.

Use "module spider" to find all possible modules and extensions.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".


[vawelke@Node-01 vawelke]$ which python
/usr/bin/python
[vawelke@Node-01 vawelke]$ NEXUS=/nexus/posix0/MIA-astro-env/myben/vawelke
[vawelke@Node-01 vawelke]$ mkdir -p $NEXUS
[vawelke@Node-01 vawelke]$ cd $NEXUS
[vawelke@Node-01 vawelke]$ curl -L -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
curl: (35) OpenSSL SSL_connect: Connection reset by peer in connection to repo.anaconda.com:443
[vawelke@Node-01 vawelke]$ wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
--2026-02-05 15:04:58--  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
Resolving repo.anaconda.com (repo.anaconda.com)... 104.16.32.241, 104.16.191.158, 2606:4700::6810:bf9e, ...
Connecting to repo.anaconda.com (repo.anaconda.com)|104.16.32.241|:443... connected.
GnuTLS: Error in the pull function.
Unable to establish SSL connection.
[vawelke@Node-01 vawelke]$ ls -d /nexus/posix0/MIA-astro-env/myben/*/miniconda3 2>/dev/null | head
[vawelke@Node-01 vawelke]$ python -m pip install --user frank
Collecting frank
  Downloading frank-1.2.3.tar.gz (81 kB)
     |████████████████████████████████| 81 kB 153 kB/s
  Preparing metadata (setup.py) ... done
Requirement already satisfied: numpy>=1.12 in /usr/lib64/python3.9/site-packages (from frank) (1.23.5)
Collecting matplotlib>=3.1.0
  Downloading matplotlib-3.9.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.3 MB)
     |████████████████████████████████| 8.3 MB 20.7 MB/s
Collecting scipy!=1.12.*,!=1.13.*,!=1.14.*,>=1.2.0
  Downloading scipy-1.11.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (36.6 MB)
     |████████████████████████████████| 36.6 MB 41 kB/s
Collecting contourpy>=1.0.1
  Downloading contourpy-1.3.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (321 kB)
     |████████████████████████████████| 321 kB 126.9 MB/s
Collecting kiwisolver>=1.3.1
  Downloading kiwisolver-1.4.7-cp39-cp39-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
     |████████████████████████████████| 1.6 MB 113.1 MB/s
Collecting fonttools>=4.22.0
  Downloading fonttools-4.60.2-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
     |████████████████████████████████| 4.8 MB 111.5 MB/s
Requirement already satisfied: pyparsing>=2.3.1 in /usr/lib/python3.9/site-packages (from matplotlib>=3.1.0->frank) (2.4.7)
Requirement already satisfied: packaging>=20.0 in /usr/lib/python3.9/site-packages (from matplotlib>=3.1.0->frank) (20.9)
Collecting pillow>=8
  Downloading pillow-11.3.0-cp39-cp39-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
     |████████████████████████████████| 6.6 MB 137.7 MB/s
Requirement already satisfied: python-dateutil>=2.7 in /usr/lib/python3.9/site-packages (from matplotlib>=3.1.0->frank) (2.9.0.post0)
Collecting importlib-resources>=3.2.0
  Downloading importlib_resources-6.5.2-py3-none-any.whl (37 kB)
Collecting cycler>=0.10
  Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Collecting zipp>=3.1.0
  Downloading zipp-3.23.0-py3-none-any.whl (10 kB)
Requirement already satisfied: six>=1.5 in /usr/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib>=3.1.0->frank) (1.15.0)
Using legacy 'setup.py install' for frank, since package 'wheel' is not installed.
Installing collected packages: zipp, pillow, kiwisolver, importlib-resources, fonttools, cycler, contourpy, scipy, matplotlib, frank
  WARNING: The scripts fonttools, pyftmerge, pyftsubset and ttx are installed in '/home/vawelke/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
^CERROR: Operation cancelled by user
[vawelke@Node-01 vawelke]$ python -m pip uninstall -y frank matplotlib scipy pillow contourpy kiwisolver fonttools cycler importlib-resources zipp
WARNING: Skipping frank as it is not installed.
WARNING: Skipping matplotlib as it is not installed.
Found existing installation: scipy 1.11.4
Uninstalling scipy-1.11.4:
  Successfully uninstalled scipy-1.11.4
Found existing installation: pillow 11.3.0
Uninstalling pillow-11.3.0:
  Successfully uninstalled pillow-11.3.0
Found existing installation: contourpy 1.3.0
Uninstalling contourpy-1.3.0:
  Successfully uninstalled contourpy-1.3.0
Found existing installation: kiwisolver 1.4.7
Uninstalling kiwisolver-1.4.7:
  Successfully uninstalled kiwisolver-1.4.7
Found existing installation: fonttools 4.60.2
Uninstalling fonttools-4.60.2:
  Successfully uninstalled fonttools-4.60.2
Found existing installation: cycler 0.12.1
Uninstalling cycler-0.12.1:
  Successfully uninstalled cycler-0.12.1
Found existing installation: importlib-resources 6.5.2
Uninstalling importlib-resources-6.5.2:
  Successfully uninstalled importlib-resources-6.5.2
Found existing installation: zipp 3.23.0
Uninstalling zipp-3.23.0:
  Successfully uninstalled zipp-3.23.0
[vawelke@Node-01 vawelke]$ NEXUS=/nexus/posix0/MIA-astro-env/myben/vawelke
[vawelke@Node-01 vawelke]$
python -m venv $NEXUS/venvs/frank_env
[vawelke@Node-01 vawelke]$ source $NEXUS/venvs/frank_env/bin/activate
(frank_env) [vawelke@Node-01 vawelke]$ pip install -U pip wheel
Requirement already satisfied: pip in ./venvs/frank_env/lib/python3.9/site-packages (21.3.1)
Collecting pip
  Downloading pip-26.0.1-py3-none-any.whl (1.8 MB)
     |████████████████████████████████| 1.8 MB 4.2 MB/s
Collecting wheel
  Downloading wheel-0.46.3-py3-none-any.whl (30 kB)
Collecting packaging>=24.0
  Downloading packaging-26.0-py3-none-any.whl (74 kB)
     |████████████████████████████████| 74 kB 703 kB/s
Installing collected packages: packaging, wheel, pip
  Attempting uninstall: pip
    Found existing installation: pip 21.3.1
    Uninstalling pip-21.3.1:
      Successfully uninstalled pip-21.3.1
Successfully installed packaging-26.0 pip-26.0.1 wheel-0.46.3
(frank_env) [vawelke@Node-01 vawelke]$ pip install git+https://github.com/discsim/frank.git
Collecting git+https://github.com/discsim/frank.git
  Cloning https://github.com/discsim/frank.git to /tmp/pip-req-build-_gh9h1jv
  Running command git clone --filter=blob:none --quiet https://github.com/discsim/frank.git /tmp/pip-req-build-_gh9h1jv
  Resolved https://github.com/discsim/frank.git to commit cdf859434a9ac9dfb1801b0fc351ff2bad74320a
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting numpy>=1.12 (from frank==1.2.3)
  Downloading numpy-2.0.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
Collecting matplotlib>=3.1.0 (from frank==1.2.3)
  Downloading matplotlib-3.9.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
Collecting scipy!=1.12.*,!=1.13.*,!=1.14.*,>=1.2.0 (from frank==1.2.3)
  Downloading scipy-1.11.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
Collecting contourpy>=1.0.1 (from matplotlib>=3.1.0->frank==1.2.3)
  Downloading contourpy-1.3.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.4 kB)
Collecting cycler>=0.10 (from matplotlib>=3.1.0->frank==1.2.3)
  Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib>=3.1.0->frank==1.2.3)
  Downloading fonttools-4.60.2-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (113 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib>=3.1.0->frank==1.2.3)
  Downloading kiwisolver-1.4.7-cp39-cp39-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (6.3 kB)
Requirement already satisfied: packaging>=20.0 in ./venvs/frank_env/lib/python3.9/site-packages (from matplotlib>=3.1.0->frank==1.2.3) (26.0)
Collecting pillow>=8 (from matplotlib>=3.1.0->frank==1.2.3)
  Downloading pillow-11.3.0-cp39-cp39-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
Collecting pyparsing>=2.3.1 (from matplotlib>=3.1.0->frank==1.2.3)
  Downloading pyparsing-3.3.2-py3-none-any.whl.metadata (5.8 kB)
Collecting python-dateutil>=2.7 (from matplotlib>=3.1.0->frank==1.2.3)
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting importlib-resources>=3.2.0 (from matplotlib>=3.1.0->frank==1.2.3)
  Downloading importlib_resources-6.5.2-py3-none-any.whl.metadata (3.9 kB)
Collecting zipp>=3.1.0 (from importlib-resources>=3.2.0->matplotlib>=3.1.0->frank==1.2.3)
  Downloading zipp-3.23.0-py3-none-any.whl.metadata (3.6 kB)
Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib>=3.1.0->frank==1.2.3)
  Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting numpy>=1.12 (from frank==1.2.3)
  Downloading numpy-1.26.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
Downloading matplotlib-3.9.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.3/8.3 MB 37.7 MB/s  0:00:00
Downloading contourpy-1.3.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (321 kB)
Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.60.2-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 23.8 MB/s  0:00:00
Downloading importlib_resources-6.5.2-py3-none-any.whl (37 kB)
Downloading kiwisolver-1.4.7-cp39-cp39-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 5.6 MB/s  0:00:00
Downloading pillow-11.3.0-cp39-cp39-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 51.3 MB/s  0:00:00
Downloading pyparsing-3.3.2-py3-none-any.whl (122 kB)
Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading scipy-1.11.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (36.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 36.6/36.6 MB 52.3 MB/s  0:00:00
Downloading numpy-1.26.4-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 69.0 MB/s  0:00:00
Downloading six-1.17.0-py2.py3-none-any.whl (11 kB)
Downloading zipp-3.23.0-py3-none-any.whl (10 kB)
Building wheels for collected packages: frank
  Building wheel for frank (pyproject.toml) ... done
  Created wheel for frank: filename=frank-1.2.3-py3-none-any.whl size=88916 sha256=49441a8e63a019f302feabe7c267586816fdae2fda939907cb4fc4684a08298b
  Stored in directory: /tmp/pip-ephem-wheel-cache-gz1ty7jv/wheels/d0/ff/a1/6b0dd6b585b50975f88bcf2e62f3c8f464f0db67062bd935a1
Successfully built frank
Installing collected packages: zipp, six, pyparsing, pillow, numpy, kiwisolver, fonttools, cycler, scipy, python-dateutil, importlib-resources, contourpy, matplotlib, frank
Successfully installed contourpy-1.3.0 cycler-0.12.1 fonttools-4.60.2 frank-1.2.3 importlib-resources-6.5.2 kiwisolver-1.4.7 matplotlib-3.9.4 numpy-1.26.4 pillow-11.3.0 pyparsing-3.3.2 python-dateutil-2.9.0.post0 scipy-1.11.4 six-1.17.0 zipp-3.23.0
(frank_env) [vawelke@Node-01 vawelke]$ python -c "import frank; print(frank.__file__)"
/nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/lib64/python3.9/site-packages/frank/__init__.py
(frank_env) [vawelke@Node-01 vawelke]$
```
