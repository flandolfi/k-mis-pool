#!/bin/bash

RESULTS_DIR="./results/"
mkdir -p $RESULTS_DIR

for DS in PATTERN CLUSTER; do
    for MODEL in GCN ChebNet GraphSAGE; do
        for PARAMS in 100K 500K; do
            for ORDER in "random" "min-local-variation"; do
                NAME="${MODEL}_P_${PARAMS}"
                python -m benchmark train --model $NAME --dataset $DS \
                    --module__ordering $ORDER \
                    --save_params ${RESULTS_DIR}/${NAME}_${ORDER}.pt \
                    --save_history ${RESULTS_DIR}/${NAME}_${ORDER}.json \
                    --batch_size 128 --num_workers 8 $@
            done
        done
    done
done
