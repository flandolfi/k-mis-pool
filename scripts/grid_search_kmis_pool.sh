#!/bin/bash

RESULTS_DIR="./results/"
CONFIG_ALL="--root datasets --local_dir $RESULTS_DIR --cpu_per_trial 4 --gpu_per_trial 0.5"
GS_CMD="python -m benchmark grid_search $CONFIG_ALL"

clean_up () {
  PREFIX=$RESULTS_DIR/${1}_${2}
  mv $PREFIX/model_assessment.json $RESULTS_DIR/${1}Mean_${2}_results.json
  mv $PREFIX/best_config.json $RESULTS_DIR/${1}Mean_${2}_config.json
  rm -r $PREFIX
}

for MODEL in KMISPoolLinear KMISPoolRandom KMISPoolNorm KMISPoolConst; do
  for DATASET in "DD" "REDDIT-BINARY" "REDDIT-MULTI-5K" "REDDIT-MULTI-12K" "github_stargazers" "MalNetTiny"; do
    $GS_CMD --dataset $DATASET --model $MODEL --opt_grid "{'k':[1,2,3],'reduce_x':['mean']}"

    clean_up $MODEL $DATASET
  done
done
