#!/bin/bash

DATASET=$1
DIR=./results/${DATASET}/

shift
mkdir -p $DIR

for ORDER in 'max-s-paths' 'max-p-paths' 'max-triangles' 'random'; do
    python -m benchmark.cv -d $DATASET -o $ORDER --to_pickle ${DIR}/${DATASET}_${ORDER}.pickle $@
done