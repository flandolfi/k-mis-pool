#!/bin/bash

RESULTS_DIR="./results/"
mkdir -p $RESULTS_DIR

for DS in MNIST CIFAR10 PATTERN CLUSTER; do
    DIR="${RESULTS_DIR}/${DS}/"
    mkdir -p $DIR

    for MODEL in GCN ChebNet GraphSAGE; do
        for PARAMS in 100K 500K; do
            for ORDER in "random" "min-local-variation"; do
                NAME="${MODEL}_P_${PARAMS}"

                python -m benchmark train --model $NAME --dataset $DS \
                    --module__ordering $ORDER \
                    --save_params ${DIR}/${NAME}.pt \
                    --save_history ${DIR}/${NAME}.json \
                    --batch_size 128 --num_workers 8 $@
            done
        done
    done
done
