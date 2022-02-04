#!/bin/bash

CMD="python -m benchmark profile --device 'cuda'"
OUT_DIR="./results/stats/"
mkdir -p $OUT_DIR

KS="'[1,2,4,8]'"
GROUP="SNAP"
for DS in com-Friendster com-Orkut com-Youtube com-LiveJournal as-Skitter; do
    $CMD --name $DS --group $GROUP --k $KS --device --output_json $OUT_DIR/${DS}.json
done

GROUP="DIMACS10"
for DS in coPapersDBLP coPapersCiteseer; do
    $CMD --name $DS --group $GROUP --k $KS --device --output_json $OUT_DIR/${DS}.json
done

KS="'[1,10,100,1000]'"
for DS in europe_osm asia_osm italy_osm; do
    $CMD --name $DS --group $GROUP --k $KS --device --output_json $OUT_DIR/${DS}.json
done
