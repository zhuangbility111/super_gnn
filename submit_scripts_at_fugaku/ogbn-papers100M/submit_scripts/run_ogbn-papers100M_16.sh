#!/bin/bash

#PJM -L "rscunit=rscunit_ft01,rscgrp=small-s2"
#PJM -L elapse=10:00:00 
#PJM -g ra000012 
#PJM -L "node=2x1x2:torus:strict-io,freq=2000"
#PJM --mpi "proc=16"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
graph_name=ogbn-papers100M
dir_stdout=../log/barrier_with_no_asnyc/${PJM_MPI_PROC}proc/
tcmalloc_path=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so
date
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori/stdout -stderr-proc ${dir_stdout}/ori/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=32
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre/stdout -stderr-proc ${dir_stdout}/ori+pre/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=true
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+int2/stdout -stderr-proc ${dir_stdout}/ori+int2/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=2
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre+int2/stdout -stderr-proc ${dir_stdout}/ori+pre+int2/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=2 --is_pre_delay=true
date
