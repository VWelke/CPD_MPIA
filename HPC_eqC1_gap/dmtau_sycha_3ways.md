# eqC1_gap — DM_Tau_gap1 (gap/) and SY_Cha_gap0 (inner_cavity/)
rsync -av --progress /mnt/d/CPD_MPIA/HPC_eqC1_gap/gap/DM_Tau_gap1/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/DM_Tau_gap1/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_eqC1_gap/gap/SY_Cha_gap0/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/SY_Cha_gap0/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_eqC1_gap/diskdictionary_eqC1.py astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionary_r90/diskdictionary_eqC1.py

# huang_gap — DM_Tau_gap1 (gap/) and SY_Cha_gap0 (inner_cavity/)
rsync -av --progress /mnt/d/CPD_MPIA/HPC_huang_gap/gap/DM_Tau_gap1/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/DM_Tau_gap1/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_huang_gap/gap/SY_Cha_gap0/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/SY_Cha_gap0/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_huang_gap/diskdictionary_newgap.py astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/diskdictionary_r90/diskdictionary_newgap.py

# r0_5_rout — DM_Tau_gap1 and SY_Cha_gap0
rsync -av --progress /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/r0_5_rout/DM_Tau_gap1/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5_rout/DM_Tau_gap1/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/r0_5_rout/SY_Cha_gap0/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5_rout/SY_Cha_gap0/


nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/DM_Tau_gap1/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/DM_Tau_gap1/pipeline.log 2>&1 &
nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/SY_Cha_gap0/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/SY_Cha_gap0/pipeline.log 2>&1 &
nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/DM_Tau_gap1/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/DM_Tau_gap1/pipeline.log 2>&1 &
nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/SY_Cha_gap0/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/SY_Cha_gap0/pipeline.log 2>&1 &
nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5_rout/DM_Tau_gap1/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5_rout/DM_Tau_gap1/pipeline.log 2>&1 &
nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5_rout/SY_Cha_gap0/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/r0_5_rout/SY_Cha_gap0/pipeline.log 2>&1 &



# rm0_5 — DM_Tau_gap1 and SY_Cha_gap0  (robust=-0.5, n_mocks=50, diskdictionaryrm0_5)
rsync -av --progress /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/rm0_5/DM_Tau_gap1/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/DM_Tau_gap1/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/rm0_5/SY_Cha_gap0/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/SY_Cha_gap0/

nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/DM_Tau_gap1/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/DM_Tau_gap1/pipeline.log 2>&1 &
nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/SY_Cha_gap0/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/SY_Cha_gap0/pipeline.log 2>&1 &



## check (all pipelines)
for folder in eqC1_gap/gap/DM_Tau_gap1 eqC1_gap/gap/SY_Cha_gap0 huang_gap/gap/DM_Tau_gap1 huang_gap/gap/SY_Cha_gap0 r0_5_rout/DM_Tau_gap1 r0_5_rout/SY_Cha_gap0 rm0_5/DM_Tau_gap1 rm0_5/SY_Cha_gap0; do echo "--- $folder ---"; ls /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/${folder}/injections/ 2>/dev/null | head -3; tail -3 /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/${folder}/pipeline.log 2>/dev/null; echo ""; done






# Upload fixed scripts
rsync -av --progress /mnt/d/CPD_MPIA/HPC_eqC1_gap/gap/DM_Tau_gap1/run_pipeline.sh astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/DM_Tau_gap1/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_huang_gap/gap/DM_Tau_gap1/run_pipeline.sh astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/DM_Tau_gap1/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_eqC1_gap/gap/SY_Cha_gap0/run_pipeline.sh astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/SY_Cha_gap0/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_huang_gap/gap/SY_Cha_gap0/run_pipeline.sh astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/SY_Cha_gap0/


# Clean up the botched outputs before rerunning
# DM_Tau_gap1 — nothing actually ran so just clear the logs
ssh astronode1 "rm -f /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/DM_Tau_gap1/{inject.log,prepimaging.log,imageloop.log,recover_loop.log,pipeline.log} /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/DM_Tau_gap1/*.custom.mask"
ssh astronode1 "rm -f /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/DM_Tau_gap1/{inject.log,prepimaging.log,imageloop.log,recover_loop.log,pipeline.log} /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/DM_Tau_gap1/*.custom.mask"
ssh astronode1 "rm -f /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/SY_Cha_gap0/{inject.log,prepimaging.log,imageloop.log,recover_loop.log,pipeline.log} /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/SY_Cha_gap0/*.custom.mask"
ssh astronode1 "rm -f /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/SY_Cha_gap0/{inject.log,prepimaging.log,imageloop.log,recover_loop.log,pipeline.log} /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/SY_Cha_gap0/*.custom.mask"

"


nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/DM_Tau_gap1/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/DM_Tau_gap1/pipeline.log 2>&1 &
nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/SY_Cha_gap0/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/SY_Cha_gap0/pipeline.log 2>&1 &
nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/DM_Tau_gap1/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/DM_Tau_gap1/pipeline.log 2>&1 &
nohup bash /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/SY_Cha_gap0/run_pipeline.sh > /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/SY_Cha_gap0/pipeline.log 2>&1 &



rsync -av --progress /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/rm0_5/SY_Cha_gap0/SY_Cha_gap0_injectloop.py astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/SY_Cha_gap0/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/rm0_5/DM_Tau_gap1/DM_Tau_gap1_injectloop.py astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/DM_Tau_gap1/
ssh astronode1 "for d in /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/SY_Cha_gap0 /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/DM_Tau_gap1; do rm -f \$d/{inject.log,prepimaging.log,imageloop.log,recover_loop.log,pipeline.log} \$d/*.custom.mask; done"


rsync -av --progress /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/rm0_5/DM_Tau_gap1/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/DM_Tau_gap1/
rsync -av --progress /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/rm0_5/SY_Cha_gap0/ astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/SY_Cha_gap0/





rsync -av --progress astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/DM_Tau_gap1/recoveries/ /mnt/d/CPD_MPIA/HPC_eqC1_gap/gap/DM_Tau_gap1/recoveries/
rsync -av --progress astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap/SY_Cha_gap0/recoveries/ /mnt/d/CPD_MPIA/HPC_eqC1_gap/gap/SY_Cha_gap0/recoveries/
rsync -av --progress astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/inner_cavity/SY_Cha_gap0/recoveries/ /mnt/d/CPD_MPIA/HPC_eqC1_gap/inner_cavity/SY_Cha_gap0/recoveries/
rsync -av --progress astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/DM_Tau_gap1/recoveries/ /mnt/d/CPD_MPIA/HPC_huang_gap/gap/DM_Tau_gap1/recoveries/
rsync -av --progress astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/huang_gap/gap/SY_Cha_gap0/recoveries/ /mnt/d/CPD_MPIA/HPC_huang_gap/gap/SY_Cha_gap0/recoveries/
rsync -av --progress astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/DM_Tau_gap1/recoveries/ /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/rm0_5/DM_Tau_gap1/recoveries/
rsync -av --progress astronode1:/nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/rm0_5/SY_Cha_gap0/recoveries/ /mnt/d/CPD_MPIA/HPC_scripts/inj_rev/rm0_5/SY_Cha_gap0/recoveries/
