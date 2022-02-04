#!/bin/bash

CMD="python -m benchmark mnist"
OUT_DIR=results/mnist/

mkdir -p $OUT_DIR
K_MAX=4

for K in `seq 0 $K_MAX`; do
  echo -n "Creating MNSIT ${K}-reduction... "
  $CMD --k $K --save_fig $OUT_DIR/mnist_L${K}.pdf
  $CMD --k $K --scorer None --ordering greedy --save_fig $OUT_DIR/mnist_C${K}.pdf
  echo "Done."
done | tqdm --null --total $(( $K_MAX + 1 ))
