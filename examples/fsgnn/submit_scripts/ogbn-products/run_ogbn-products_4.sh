#!/bin/bash

#PJM -L "rscunit=rscunit_ft01,rscgrp=small-s2"
#PJM -L elapse=8:00:00 
#PJM -g ra000012 
#PJM -L "node=1x1x1:torus:strict-io,freq=2000"
#PJM --mpi "proc=4"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
graph_name=ogbn-products
dir_stdout=./log/${PJM_MPI_PROC}proc/
tcmalloc_path=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so
date
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/stdout -stderr-proc ${dir_stdout}/stderr python ../../train.py --config=../../config/${graph_name}.yaml --num_bits=32
date
