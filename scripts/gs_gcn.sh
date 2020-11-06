#!/bin/bash

# Example:
#    sh ./gs_gcn.sh MNIST min-curvature

GNN=GCN
DIR=./results/
DS=$1
ORDER=$2

mkdir -p $DIR

python -m benchmarks.benchmark grid_search --model_name=GCN --dataset_name=$DS \
    --cv_results_path=$DIR/${GNN}_L2_${DS}_${ORDER}_GS.csv --batch_size=128 --lr=0.001 \
    --module__blocks=2 --module__hidden=104 --module__incremental=False --module__ordering=$ORDER $@
python -m benchmarks.benchmark grid_search --model_name=GCN --dataset_name=$DS \
    --cv_results_path=$DIR/${GNN}_L3_${DS}_${ORDER}_GS.csv --batch_size=128 --lr=0.001 \
    --module__blocks=3 --module__hidden=85 --module__incremental=False --module__ordering=$ORDER $@
