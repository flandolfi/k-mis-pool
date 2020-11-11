#!/bin/bash

# Example:
#    sh ./cv_gcn.sh MNIST

GNN=GCN
DIR=./results/
DS=$1

OPTS="--model_name=$GNN --dataset_name=$DS --batch_size=128" \
     "--repetitions=4 --module__aggr=add --module__normalize"

mkdir -p $DIR
shift

for ORDER in random max-norm max-curvature min-curvature min-degree; do
    python -m benchmark.benchmarks cv $OPTS \
        --params_path=$DIR/${GNN}_L2_${DS}_${ORDER}_PARAMS.pt \
        --cv_results_path=$DIR/${GNN}_L2_${DS}_${ORDER}_CV.csv \
        --module__blocks=2 --module__hidden=106 --module__ordering=$ORDER $@
    python -m benchmark.benchmarks cv $OPTS \
        --params_path=$DIR/${GNN}_L2_${DS}_${ORDER}_PARAMS.pt \
        --cv_results_path=$DIR/${GNN}_L2_${DS}_${ORDER}_CV.csv \
        --module__blocks=2 --module__hidden=198 --module__ordering=$ORDER $@
done
