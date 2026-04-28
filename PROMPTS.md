# PROMPTS.md

---

# 2026-04-27

## Kink File Prep

```text
Prompt: The HPC_scripts contain directories for running injection recovery experiments on different disk gap locations, I want to now run the same pipeline on kink locations, please read the HPC_scritps directories and let me know what i need to prepare , eg , the differnt disk dictionary?  change the individual script keywords? and what else
```

```text
Reply:   Recommended order of work

  1. Determine kink radii/widths from your detection analysis → fill rkink/wkink
  2. Update all disk dictionary files
  3. Patch custom_mask.py (one-time shared-code change)
  4. Create one test directory (e.g. AA_Tau_kink0/) with modified scripts
  5. Run the inject → preimage → image → recover chain on that one disk end-to-end
  6. Once validated, create the remaining {Disk}_kink{N}/ directories
```

```text
Prompt: Please help me find the kink radii as they and their properties are defined somewhere in the directory, add them to diskdictionary.ipynb where I defined scripts to gernerate diskdictionary.py s ,  add wkink to be a fix constant first , I'll look at them later. dont update all the disk dictionary files. Create one test directory please.
```

---
