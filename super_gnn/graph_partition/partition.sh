#!/bin/bash

date

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhuangb/super_gnn/super_gnn/graph_partition/Metis/local/lib
export PATH=$PATH:/home/zhuangb/super_gnn/super_gnn/graph_partition/Metis/local/bin

graph_name=ogbn-papers100M
python preprocess_graph.py --graph_name=${graph_name} --dataset=${graph_name} --in_dir=./dataset/ 
# python postprocess_graph_multi_proc.py -o ./ogbn_papers100M_8192_part -ir ./ -ip ./ogbn_papers100M_8192_part -g ogbn-papers100M -b 0 -e 8192 -p 8

num_procs=4

for ((num_part_each_proc=4096; num_part_each_proc<=4096; num_part_each_proc*=2))
do
	total_num_procs=$((num_procs * num_part_each_proc))
	echo "split ${graph_name} into ${total_num_procs} parts ..."
	out_dir=ogbn_${graph_name}_${total_num_procs}_part
	mkdir ${out_dir}
	mpirun -np $num_procs pm_dglpart ogbn-papers100M $num_part_each_proc
	mv p[0-9]*.txt ${out_dir}

    echo "postprocess ${graph_name} ..."
    python postprocess_graph_multi_proc.py -o ${out_dir} -ir ./ -ip ${out_dir} -g ogbn-papers100M -b 0 -e ${total_num_procs} -p 8
done

# for ((total_num_procs=512; total_num_procs<=1024; total_num_procs*=2))
# do
# 	out_dir=ogbn_${graph_name}_${total_num_procs}_part
# 	python postprocess_graph_multi_proc.py --out_dir=${out_dir} --in_dir=./ --graph_name=${graph_name} --begin_partition=0 --end_partition=${total_num_procs} --num_process=8
# done
date
