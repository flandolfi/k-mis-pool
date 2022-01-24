#!/bin/bash

CONFIG_ALL="--root datasets --cpu_per_trial 4 --gpu_per_trial 0.25"
GS_CMD="python -m benchmark grid_search $CONFIG_ALL"

for DATASET in "DD" "REDDIT-BINARY" "REDDIT-MULTI-5K" "REDDIT-MULTI-12K"; do
  for MODEL in Baseline GraclusPool; do
    $GS_CMD --dataset $DATASET --model $MODEL
  done

  for MODEL in Baseline TopKPool SAGPool ASAPool; do
    $GS_CMD --dataset $DATASET --model $MODEL --opt_grid "{'ratio':[0.5,0.2,0.1]}"
  done

  for MODEL in KMISPool KMISPoolRandom KMISPoolNorm; do
    $GS_CMD --dataset $DATASET --model $MODEL --opt_grid "{'k':[1,2,3]}"
  done

  $GS_CMD --dataset $DATASET --model PANPool \
    --opt_grid "{'ratio':[0.5,0.2,0.1],'filter_size':[1,2,3],'gnn_class':['PANConv']}"
done
