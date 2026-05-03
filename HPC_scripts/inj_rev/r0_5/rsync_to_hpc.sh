#!/usr/bin/env bash
# Upload all r0_5 disk/gap folders to HPC

rsync -av --progress \
  /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/r0_5/ \
  astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5/

# Also push updated diskdictionary_r90 if needed
# rsync -av --progress \
#   /mnt/d/CPD_MPIA/HPC_scripts/Source_codes/diskdictionary_r90/ \
#   astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionary_r90/
