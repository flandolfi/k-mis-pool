#!/bin/bash

CMD="python -m benchmark weight --root datasets --device cuda:0"
OUT_DIR=results/weights/

mkdir -p $OUT_DIR

for ORD in greedy div-k-degree div-k-sum; do
  for GRAPH in ca-AstroPh email-Enron loc-Brightkite; do
    for K in `seq 1 3`; do
      $CMD --ordering $ORD --name $GRAPH --group SNAP --k $K --store_results $OUT_DIR/${GRAPH}_${ORD}_${K}.json
    done
  done

  GRAPH=luxembourg_osm

  for K in `seq 1 8`; do
    $CMD --ordering $ORD --name $GRAPH --group DIMACS10 --k $K --store_results $OUT_DIR/${GRAPH}_${ORD}_${K}.json
  done
done