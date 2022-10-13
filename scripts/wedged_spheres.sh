#!/bin/sh

./es.sh WEDGED-SPHERES-2 "poetry run python ../toast/cli.py -r 0.05 -R 0.25 -s 0.1 -S 0.5 -d 2 --num-steps 20 ../data/Wedged_spheres_2D.txt > ../output/Wedged_spheres_2D.txt"
