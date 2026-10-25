#!/usr/bin/env bash
# Combine the base run + rep1-9 of each HD_135344B uvfix candidate into
# recoveries/HD_135344B_gap{N}_recoveries.combined.txt (10 runs x 50 = 500 mocks
# per flux bin), then run assess_recovery.py in the base folder (figures -> assess_figs/).
#
# usage (on Node-01):  bash combine_and_assess_HD135344B.sh            # C1, C2 and DF
#                      bash combine_and_assess_HD135344B.sh C2:3       # just one, as candidate:gap_ix

cd /nexus/posix0/MIA-astro-env/myben/vawelke/inj_rev/eqC1_gap/gap
source /nexus/posix0/MIA-astro-env/myben/vawelke/venvs/frank_env/bin/activate

for p in ${@:-C1:2 C2:3 DF:4}; do
    c=${p%:*}; g=${p#*:}
    d=HD_135344B_${c}_uvfix
    f=recoveries/HD_135344B_gap${g}_recoveries

    cat $d/$f.0.txt ${d}_rep[1-9]/$f.0.txt > $d/$f.combined.txt
    echo "=== $d: $(ls $d/$f.0.txt ${d}_rep[1-9]/$f.0.txt | wc -l) runs combined, mocks per flux bin:"
    awk '{n[$1]++} END {for (F in n) print "  F = " F " uJy : " n[F]}' $d/$f.combined.txt | sort -k3 -n

    (cd $d && python assess_recovery.py > assess.log 2>&1)
    echo "  assess_recovery.py done (log: $d/assess.log), figures:"
    ls $d/assess_figs | sed 's/^/    /'
done

deactivate
