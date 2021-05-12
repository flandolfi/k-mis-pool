#!/bin/bash

RESULTS_DIR="./results/"
mkdir -p $RESULTS_DIR

for DS in MNIST CIFAR10 CLUSTER PATTERN; do
    DIR="${RESULTS_DIR}/${DS}/"
    mkdir -p $DIR

    for MODEL in GCN ChebNet GraphSAGE; do
        for CONFIG in 22 44; do
            for SEED in `seq 5`; do
                NAME="${MODEL}_${CONFIG}"

                python -m benchmark train --model $NAME --dataset $DS \
                    --save_params ${DIR}/${NAME}__${SEED}.pt \
                    --save_history ${DIR}/${NAME}__${SEED}.json \
                    --batch_size 128 --num_workers 8 --seed $SEED $@
            done
        done
    done
done
