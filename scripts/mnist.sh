#!/bin/bash

DIR="./results/img/"
mkdir -p $DIR

FILE=${DIR}/original.pdf

echo "Generating $FILE"
python -m benchmark mnist --pool --k 0 --noshow --save_fig $FILE

for K in 1 2 3; do
    FILE=${DIR}/avgpool_${K}.pdf

    echo "Generating $FILE"
    python -m benchmark mnist --pool --k $K --noshow --save_fig $FILE

    for ORDER in None max-norm min-local-variation; do
        for OPT in sample_aggregate nosample_partition; do
            FILE=${DIR}/${K}-mis_${ORDER}_${OPT}.pdf

            echo "Generating $FILE"
            python -m benchmark mnist --k $K --${OPT} --noshow --save_fig $FILE
        done
    done
done