#!/bin/sh

for POINTS in 1000 2500 5000 10000; do
  ./es.sh IPSC-${POINTS} "poetry run python ../toast/cli.py ../data/ipsc.npz --seed 42 -q ${POINTS} -d 16 -o ../output/ipsc_d16_q${POINTS}_seed42.csv"
done
