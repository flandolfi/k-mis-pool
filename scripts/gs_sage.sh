#!/bin/bash

# Example:
#    sh ./gs_sage.sh MNIST min-curvature

GNN=GCN
DIR=./results/
DS=$1
ORDER=$2

mkdir -p $DIR

python -m benchmark.benchmarks grid_search --model_name=GraphSAGE --dataset_name=$DS \
    --cv_results_path=$DIR/${GNN}_L2_${DS}_${ORDER}_GS.csv --batch_size=128 --lr=0.001 \
    --module__blocks=2 --module__hidden=76 --module__incremental=False --module__ordering=$ORDER $@
python -m benchmark.benchmarks grid_search --model_name=GraphSAGE --dataset_name=$DS \
    --cv_results_path=$DIR/${GNN}_L3_${DS}_${ORDER}_GS.csv --batch_size=128 --lr=0.001 \
    --module__blocks=3 --module__hidden=62 --module__incremental=False --module__ordering=$ORDER $@
