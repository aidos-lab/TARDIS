#!/bin/sh

./es.sh        MNIST "python ../plh/cli.py -d 10 --num-steps 20        MNIST > ../output/MNIST.txt"
./es.sh FashionMNIST "python ../plh/cli.py -d 10 --num-steps 20 FashionMNIST > ../output/FashionMNIST.txt"
