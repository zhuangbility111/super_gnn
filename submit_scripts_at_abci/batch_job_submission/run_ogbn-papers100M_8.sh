#!/bin/bash

#$-l rt_F=4
#$-cwd
#$-l h_rt=10:00:00

source ~/gcn.work/dgl_intel_setting_1/env_torch_1.10.0.sh
source /etc/profile.d/modules.sh
# load mpi library
module load intel-mpi/2021.8
export FI_PROVIDER=tcp

graph_name=ogbn-papers100M
# number of total processes 
NP=8

# number of processes per node
NPP=2

tcmalloc_path=/home/aaa10008ku/gcn.work/dgl_intel_setting_1/sub407/miniconda3/envs/torch-1.10/lib/libtcmalloc.so
date
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../examples/graphsage/train.py --config=../../examples/graphsage/config/abci/${graph_name}.yaml --num_bits=32 --is_label_augment=true
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../examples/graphsage/train.py --config=../../examples/graphsage/config/abci/${graph_name}.yaml --num_bits=32 --is_pre_delay=true --is_label_augment=true
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../examples/graphsage/train.py --config=../../examples/graphsage/config/abci/${graph_name}.yaml --num_bits=2 --is_label_augment=true
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../examples/graphsage/train.py --config=../../examples/graphsage/config/abci/${graph_name}.yaml --num_bits=2 --is_pre_delay=true --is_label_augment=true
date
