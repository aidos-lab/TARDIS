#!/bin/bash
#
# es.sh: easy submit --- submit a job quickly to the cluster
#
# This script tries to provide a 'fire and forget' solution for
# submitting CPU jobs to the cluster. The parameters may not be
# perfect, but it's probably sufficient for most cases.

NAME="$1"
CMD="$2"

if [ -z "$2" ]; then
  NAME="easy-submit"
  CMD=$1
fi

if [ -z "$CMD" ]; then
  echo "Usage: $0 [NAME] COMMAND"
  echo "  Specify at least the command to run."
  exit -1
fi

MAIL_USER=${USER}@helmholtz-muenchen.de

sbatch -p cpu_p                 \
       -J ${NAME}               \
       -o "${NAME}_%j.out"      \
       --cpus-per-task=4       \
       --mem=16G                \
       --mail-type=END,FAIL     \
       --mail-user=${MAIL_USER} \
       --nice=10000             \
       --wrap "${CMD}"
