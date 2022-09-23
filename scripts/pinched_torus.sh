#!/bin/sh

for S in 0.5 0.55 0.60 0.65 0.75; do
  ./es.sh PINCHED_TORUS "python ../plh/cli.py ../data/Pinched_torus.txt -r 0.05 -R 0.45 -s 0.2 -S ${S} > ../output/Pinched_torus_S${S}.txt"
done
