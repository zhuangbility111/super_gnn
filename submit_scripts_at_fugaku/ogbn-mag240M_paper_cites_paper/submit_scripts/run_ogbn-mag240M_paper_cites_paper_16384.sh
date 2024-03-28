#!/bin/bash

#PJM -L "rscunit=rscunit_ft01,rscgrp=large"
#PJM -L elapse=01:00:00 
#PJM -g ra000012 
#PJM -L "node=16x16x16:torus:strict-io,freq=2000"
#PJM --mpi "proc=16384"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
graph_name=ogbn-mag240M_paper_cites_paper
# dir_stdout=../log/barrier_with_no_asnyc_v1/${PJM_MPI_PROC}proc/
# dir_stdout=../log/barrier_with_no_asnyc_v1/${PJM_MPI_PROC}proc/
# res_path=../log/test_label_augment+int2+pre_lr0.005
res_path=../log/hidden256+layer3+lr0.005+decay0.0+epoch300+label/
mkdir $res_path
dir_stdout=${res_path}/${PJM_MPI_PROC}proc/
tcmalloc_path=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so
model_params="--hidden_channels=256 --num_layers=3 --lr=0.005 --weight_decay=0.0 --num_epochs=300"
date
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori/stdout -stderr-proc ${dir_stdout}/ori/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=false ${model_params} --is_label_augment=true
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre/stdout -stderr-proc ${dir_stdout}/ori+pre/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=true ${model_params} --is_label_augment=true
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+int2/stdout -stderr-proc ${dir_stdout}/ori+int2/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=2 --is_pre_delay=false ${model_params} --is_label_augment=true
# LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre+int2/stdout -stderr-proc ${dir_stdout}/ori+pre+int2/stderr python ../../../examples/graphsage/train.py --config=../../../examples/graphsage/config/fugaku/${graph_name}.yaml --num_bits=2 --is_pre_delay=true ${model_params} --is_label_augment=true
date