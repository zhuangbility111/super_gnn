#!/bin/bash

#PJM -L "rscunit=rscunit_ft01,rscgrp=small-s1"
#PJM -L elapse=1:00:00 
#PJM -g ra000012 
#PJM -L "node=8x4x8:torus:strict-io,freq=2000"
#PJM --mpi "proc=1024"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
graph_name=uk-2007-02
dir_stdout=../log/weak_scaling/${PJM_MPI_PROC}proc/
tcmalloc_path=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so
date
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori/stdout -stderr-proc ${dir_stdout}/ori/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}_weak_scaling.yaml --num_bits=32
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre/stdout -stderr-proc ${dir_stdout}/ori+pre/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}_weak_scaling.yaml --num_bits=32 --is_pre_delay=true
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+int2/stdout -stderr-proc ${dir_stdout}/ori+int2/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}_weak_scaling.yaml --num_bits=2
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre+int2/stdout -stderr-proc ${dir_stdout}/ori+pre+int2/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}_weak_scaling.yaml --num_bits=2 --is_pre_delay=true
date
