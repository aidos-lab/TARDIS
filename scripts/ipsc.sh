#!/bin/sh

for NUM in $(seq 1 5); do
  ./es.sh IPSC-${NUM} "poetry run python ../toast/cli.py ../data/ipsc.npz -k 10 -d 20 --num-steps 20 > ../output/ipsc_d20_${NUM}.txt"
done
