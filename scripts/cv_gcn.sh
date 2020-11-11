#!/bin/bash

# Example:
#    sh ./cv_gcn.sh MNIST

GNN=GCN
DIR=./results/
DS=$1

mkdir -p $DIR
shift

for ORDER in random max-norm max-curvature min-curvature min-degree; do
    python -m benchmark.benchmarks cv --model_name=GCN --dataset_name=$DS \
        --cv_results_path=$DIR/${GNN}_L2_${DS}_${ORDER}_GS.csv --batch_size=128 \
        --module__aggr=add --module__normalize --module__blocks=2 \
        --module__hidden=106 --module__ordering=$ORDER $@
    python -m benchmark.benchmarks cv --model_name=GCN --dataset_name=$DS \
        --cv_results_path=$DIR/${GNN}_L3_${DS}_${ORDER}_GS.csv --batch_size=128 \
        --module__aggr=add --module__normalize --module__blocks=2 \
        --module__hidden=198 --module__ordering=$ORDER $@
done
