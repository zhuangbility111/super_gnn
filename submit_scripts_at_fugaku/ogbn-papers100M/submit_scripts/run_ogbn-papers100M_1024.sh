#!/bin/bash

#PJM -L "rscunit=rscunit_ft01,rscgrp=small-s1"
#PJM -L elapse=01:00:00
#PJM -g ra000012 
#PJM -L "node=8x4x8:torus:strict-io,freq=2000"
#PJM --mpi "proc=1024"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
graph_name=ogbn-papers100M
res_path=../log/test_label_augment+int2+pre_lr0.003
# mkdir $res_path
dir_stdout=${res_path}/${PJM_MPI_PROC}proc/
tcmalloc_path=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so
# key=d0eb705134b7d7928895feea0ad74b3fccc6e11f
date
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori/stdout -stderr-proc ${dir_stdout}/ori/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=false --is_label_augment=true
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre/stdout -stderr-proc ${dir_stdout}/ori+pre/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=true --is_label_augment=true
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+int2/stdout -stderr-proc ${dir_stdout}/ori+int2/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=2 --is_pre_delay=false --is_label_augment=true
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre+int2/stdout -stderr-proc ${dir_stdout}/ori+pre+int2/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=2 --is_pre_delay=true --is_label_augment=true

# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+int8/stdout -stderr-proc ${dir_stdout}/ori+int8/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=8 --is_pre_delay=false
# WANDB_API_KEY=$key LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori/stdout -stderr-proc ${dir_stdout}/ori/stderr python ../../../examples/graphsage/train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=false
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre/stdout -stderr-proc ${dir_stdout}/ori+pre/stderr python ../../../examples/graphsage/train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=true --lr=0.01
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+int2/stdout -stderr-proc ${dir_stdout}/ori+int2/stderr python ../../../examples/graphsage/train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=2
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre+int2/stdout -stderr-proc ${dir_stdout}/ori+pre+int2/stderr python ../../../examples/graphsage/train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=2 --is_pre_delay=true
date
