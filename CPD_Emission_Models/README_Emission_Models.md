This folder stores method to code the various emission models of CPD.

1. Zhu et al 2018  : parametric viscous disk model
2. Choksi and Ciang: SED model for blackbody accretion shocks + circumplanetary spheres/disks with reprocessing of radiation
3. Taylor and Adams: semi-analytical CPD + CPE with envelope temperature estimation from 2 methods.
4. Szulagyi 2025 - PDS 70c  

! Technically, if they computed an SED or provided the function for it, as both the flux and the frequency is known..
How to get one single flux value for a frequency? 

Direction might be: input temperature + density profile -> RADMC3D to get SED ->
another thing: carta measure noise at emissio free region for all disk+robust val. use SNR Map to source detect , cross check with carta and catalog and only plot emission free region