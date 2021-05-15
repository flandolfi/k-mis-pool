#!/bin/bash

RESULTS_DIR="./results/"
mkdir -p $RESULTS_DIR

for DS in MNIST CIFAR10 CLUSTER PATTERN; do
    DIR="${RESULTS_DIR}/${DS}/"
    mkdir -p $DIR

    for MODEL in GCN ChebNet GraphSAGE; do
        for K in 0 1 2; do
            for SEED in `seq 5`; do
                NAME="${MODEL}_44"

                python -m benchmark train --model $NAME --dataset $DS \
                    --save_params ${DIR}/${NAME}_K${K}__${SEED}.pt \
                    --save_history ${DIR}/${NAME}_K${K}__${SEED}.json \
                    --module__k $K --batch_size 128 --num_workers 8 \
                    --seed $SEED $@
            done
        done
    done
done
