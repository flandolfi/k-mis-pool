#!/bin/bash

DIR="./results/"
echo "dataset,model,accuracy"

for FILE in `ls ${DIR}/*/*.pt`; do
    NAME="${FILE%__*}"
    NAME="${NAME#$DIR/*}"
    MODEL="${NAME##*/}"
    DS="${NAME%/*}"

    echo -n "${DS},${MODEL},"
    python -m benchmark score --model $MODEL --dataset $DS --params_path $FILE \
        --batch_size 128 $@ | tail -n 1
done
