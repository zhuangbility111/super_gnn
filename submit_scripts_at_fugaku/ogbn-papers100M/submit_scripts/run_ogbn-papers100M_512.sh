#!/bin/bash

#PJM -L "rscunit=rscunit_ft01,rscgrp=small-s1"
#PJM -L elapse=1:00:00 
#PJM -g ra000012 
#PJM -L "node=4x4x8:torus:strict-io,freq=2000"
#PJM --mpi "proc=512"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
graph_name=ogbn-papers100M
res_path=../log/test_label_augment
# mkdir $res_path
dir_stdout=${res_path}/${PJM_MPI_PROC}proc/
tcmalloc_path=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so
# key=d0eb705134b7d7928895feea0ad74b3fccc6e11f
date
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori/stdout -stderr-proc ${dir_stdout}/ori/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=false
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+int2/stdout -stderr-proc ${dir_stdout}/ori+int2/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=2 --is_pre_delay=false
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori/stdout -stderr-proc ${dir_stdout}/ori/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=32
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre/stdout -stderr-proc ${dir_stdout}/ori+pre/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=true
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+int2/stdout -stderr-proc ${dir_stdout}/ori+int2/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=2
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre+int2/stdout -stderr-proc ${dir_stdout}/ori+pre+int2/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=2 --is_pre_delay=true
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori/stdout -stderr-proc ${dir_stdout}/ori/stderr python ../../../examples/graphsage/train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=false
date
