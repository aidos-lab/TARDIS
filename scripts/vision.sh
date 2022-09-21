#!/bin/sh

for NUM in $(seq 1 5); do
  ./es.sh        MNIST "python ../plh/cli.py -d 10 --num-steps 20 MNIST > ../output/MNIST_${NUM}.txt"
  ./es.sh FashionMNIST "python ../plh/cli.py -d 10 --num-steps 20 FashionMNIST > ../output/FashionMNIST_${NUM}.txt"
done

