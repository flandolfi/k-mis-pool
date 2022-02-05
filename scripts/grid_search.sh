#!/bin/bash

RESULTS_DIR="./results/classification/"
CONFIG_ALL="--root datasets --local_dir $RESULTS_DIR --cpu_per_trial 4 --gpu_per_trial 0.5"
GS_CMD="python -m benchmark grid_search $CONFIG_ALL"

clean_up () {
  PREFIX=$RESULTS_DIR/${1}_${2}
  mv $PREFIX/model_assessment.json ${PREFIX}_results.json
  mv $PREFIX/best_config.json ${PREFIX}_config.json
  rm -r $PREFIX
}

for DATASET in "DD" "REDDIT-BINARY" "REDDIT-MULTI-5K" "REDDIT-MULTI-12K" "github_stargazers"; do
  for MODEL in Baseline GraclusPool; do
    $GS_CMD --dataset $DATASET --model $MODEL

    clean_up $MODEL $DATASET
  done

  for MODEL in TopKPool SAGPool ASAPool; do
    $GS_CMD --dataset $DATASET --model $MODEL --opt_grid "{'ratio':[0.5,0.2,0.1]}"

    clean_up $MODEL $DATASET
  done

  for MODEL in KMISPool KMISPoolRandom KMISPoolConst KMISPoolNorm; do
    $GS_CMD --dataset $DATASET --model $MODEL --opt_grid "{'k':[1,2,3]}"

    clean_up $MODEL $DATASET
  done

  MODEL=PANPool
  $GS_CMD --dataset $DATASET --model $MODEL \
    --opt_grid "{'ratio':[0.5,0.2,0.1],'filter_size':[1,2,3],'gnn_class':['PANConv']}"

  clean_up $MODEL $DATASET
done
