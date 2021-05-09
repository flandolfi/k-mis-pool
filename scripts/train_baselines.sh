#!/bin/bash

RESULTS_DIR="./results/"
mkdir -p $RESULTS_DIR

for DS in MNIST CIFAR10 CLUSTER PATTERN; do
    for MODEL in GCN ChebNet GraphSAGE; do
        for PARAMS in 100K 500K; do
            NAME="${MODEL}_${PARAMS}"
            python -m benchmark train --model $NAME --dataset $DS \
                --save_params ${RESULTS_DIR}/${NAME}.pt \
                --save_history ${RESULTS_DIR}/${NAME}.json \
                --batch_size 128 --num_workers 8 $@
        done
    done
done
