#!/bin/sh

for NUM in $(seq 1 5); do
  ./es.sh        MNIST "poetry run python ../toast/cli.py -d 10 --num-steps 20 MNIST > ../output/MNIST_${NUM}.txt"
  ./es.sh FashionMNIST "poetry run python ../toast/cli.py -d 10 --num-steps 20 FashionMNIST > ../output/FashionMNIST_${NUM}.txt"
done

