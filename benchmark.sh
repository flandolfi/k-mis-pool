#!/bin/bash

DATASET=$1
DIR=./results/${DATASET}/

shift
mkdirs -p $DIR

for ORDER in 'min-d-paths' 'max-d-paths' 'min-s-paths' 'max-s-paths' 'min-degree' 'max-degree' 'random'; do
    python -m benchmark.cv -d $DATASET -o $ORDER --to_pickle ${DIR}/${DATASET}_${ORDER}.pickle $@
done