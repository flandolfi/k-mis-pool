#!/bin/bash

RESULTS_DIR="./results/"
mkdir -p $RESULTS_DIR

for DS in MNIST CIFAR10; do
    for MODEL in GCN ChebNet GraphSAGE; do
        for PARAMS in 100K 500K; do
            for POOL in "_" "_P_"; do
                NAME="${MODEL}${POOL}${PARAMS}"
                python -m benchmark train --model $NAME --dataset $DS \
                    --save_params ${RESULTS_DIR}/${NAME}.pt \
                    --save_history ${RESULTS_DIR}/${NAME}.json \
                    --batch_size 128 --num_workers 8 $@
            done
        done
    done
done
