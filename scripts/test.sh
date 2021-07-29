#!/bin/bash

DIR="./results/"
echo "dataset,model,k,accuracy"

for FILE in `ls ${DIR}/*/*.pt`; do
    NAME="${FILE%__*}"
    K="${NAME#*_K}"
    NAME="${NAME#$DIR/*}"
    MODEL="${NAME##*/}"
    MODEL="${MODEL%_K*}"
    DS="${NAME%/*}"

    echo -n "${DS},${MODEL%_44*},${K},"
    python -m benchmark score --model $MODEL --dataset $DS --params_path $FILE \
        --batch_size 128 $@ | tail -n 1
done
