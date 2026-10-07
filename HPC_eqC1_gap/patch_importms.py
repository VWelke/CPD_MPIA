import sys

path = "/nexus/posix0/MIA-astro-env/myben/vawelke/Source_codes/ImportMS.py"

OLD = """    # load the model visibilities
    mdl = (np.load(modelfile+'.npz'))['V']

    # replace with the model visibilities (equal in both polarizations)
    data[:, :, unflagged] = mdl"""

NEW = """    # load the model visibilities
    mdl = (np.load(modelfile+'.npz'))['V']

    # some disks' saved model/vis arrays cover ALL rows (not just the
    # currently-unflagged ones) -- if mdl's length matches the total row
    # count instead of the unflagged count, subset it the same way here so
    # the assignment below still lines up (no-op for disks with 0 flagged rows)
    if mdl.shape[-1] == unflagged.size and mdl.shape[-1] != int(unflagged.sum()):
        mdl = mdl[..., unflagged]

    # replace with the model visibilities (equal in both polarizations)
    data[:, :, unflagged] = mdl"""

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

if NEW in content:
    print("already patched, no change made")
    sys.exit(0)

if OLD not in content:
    print("ERROR: expected block not found -- file may differ from what was reviewed, aborting")
    sys.exit(1)

content = content.replace(OLD, NEW, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("patched:", path)
