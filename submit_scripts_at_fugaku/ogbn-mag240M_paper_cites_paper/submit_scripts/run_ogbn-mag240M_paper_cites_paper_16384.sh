#!/bin/bash

#PJM -L "rscunit=rscunit_ft01,rscgrp=large"
#PJM -L elapse=1:00:00 
#PJM -g ra000012 
#PJM -L "node=16x16x16:torus:strict-io,freq=2000"
#PJM --mpi "proc=16384"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
graph_name=ogbn-mag240M_paper_cites_paper
res_dir=../log/perf_check_breakdown
mkdir -p ${res_dir}
dir_stdout=${res_dir}/${PJM_MPI_PROC}proc/

tcmalloc_path=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so
date
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori/stdout -stderr-proc ${dir_stdout}/ori/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=32 --is_label_augment=true
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre/stdout -stderr-proc ${dir_stdout}/ori+pre/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=true --is_label_augment=true
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+int2/stdout -stderr-proc ${dir_stdout}/ori+int2/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=2 --is_label_augment=true
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre+int2/stdout -stderr-proc ${dir_stdout}/ori+pre+int2/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=2 --is_pre_delay=true --is_label_augment=true
date
