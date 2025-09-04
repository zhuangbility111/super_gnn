# SuperGNN: Scaling Large-scale GNN Training to Thousands of Processors on CPU-based Supercomputers (ICS'25)
This code is for paper **Scaling Large-scale GNN Training to Thousands of Processors on CPU-based Supercomputers** accepted by ICS 2025. Link of Paper: https://dl.acm.org/doi/10.1145/3721145.3730422
## Prerequisite
* PyTorch (version >= 1.10.0)
* numpy
* pandas
## Installation
1. Install pytorch-sparse: 
	1. `git clone [link in my github repo]`
	3. `cd pytorch-sparse && git checkout a64fx_dev && python setup.py install`
2. Install pytorch-scatter:
	1. `git clone [link in my github repo]`
	2. `cd pytorch-scatter && git checkout a64fx_dev && python setup.py install`
3. Install pytorch-geometric:
	1. `git clone [link in my github repo]`
	2. `cd pytorch-geometric && git checkout zhuang_dev && python setup.py install`
4. Install ParMetis for graph partition
	1. please follow the instructions provided by subsection *ParMETIS Installation* in the https://docs.dgl.ai/en/0.9.x/guide/distributed-partition.html# 
5. Install this framework:
	1. `git clone [link]`
	2. Install kernel
		1. `cd super_gnn/ops && python setup.py install`
	3. Install framework
		1. `cd ../../ && python setup.py install`
## Run full-batch graphsage training
### 1. Graph partition
1. Preprocess raw graph data:
	1. `cd super_gnn/graph_partition/`
	2. `python preprocess_graph.py --dataset=${graph_name} --raw_dir=./dataset/  --processed_dir=${processed_dir} --is_undirected`
		- `--dataset` is the name of dataset, option: \[ogbn-arxiv, ogbn-products, reddit, proteins, ogbn-papers100M, ogbn-mag240M\]
		- `--raw_dir` is the root directory for saving the raw dataset
		- `--processed_dir` is the directory for saving the preprocessed dataset that will be used by later graph partition
		- `--is_undirected` make the raw graph dataset to be undirected (for directed graph)
2. Partition graph with ParMetis
	1. `cd ${processed_dir}`
	2. make sure you have set two environment variables for ParMetis:
		- `export PATH=$PATH:$HOME/local/bin`
		- `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/local/lib/`
	3. `mpirun -np ${num_procs} pm_dglpart ${graph_name} ${num_part_each_proc}`
		1. `num_procs` is the number of MPI processes for graph partition
		2. `num_part_each_proc` is the number of subgraphs each MPI processes generates. So the total number of subgraph after graph partition is `num_procs` \* `num_part_each_proc`
		3. `graph_name` is the name of dataset, option: \[ogbn_arxiv, ogbn_products, reddit, proteins, ogbn_papers100M, ogbn_mag240M_paper_cites_paper\]
	4. With the previous 3 steps, you will get the graph partition results in \${processed_dir}. 
3. Postprocess graph partition result
	1. Back to directory `super_gnn/graph_partition/`, `python postprocess_graph_multi_proc.py -o ${out_dir} -ir ${in_raw_dir} -ip ${in_partition_dir} -g ${graph_name} -b 0 -e ${total_num_procs} -p ${num_process}`
		1. `out_dir`: the directory for saving the postprocessed dataset. **the name of this folder must to be set as**: \${graph_name}\_\${num_of_subgraphs}\_part/
		2. `in_raw_dir`: the directory of saving the preprocessed dataset
		3. `in_partition_dir`: the directory of saving the graph partition results (\${processed_dir})
		4. `graph_name` is the name of dataset, option: \[ogbn_arxiv, ogbn_products, reddit, proteins, ogbn_papers100M, ogbn_mag240M_paper_cites_paper\]
		5. `b`: is the id of beginning subgraphs, default: 0
		6. `e`: is the id of ending subgraphs, default: `total_num_subgraphs`
		7. `num_process`: the number of processes spawned for postprocessing
4. With the previous 3 steps, graph partitioning for getting a specific number of subgraphs is over. if you want to get the other number of subgraphs, you need to repeat step 2 and step 3 for graph partition, change the value of `num_procs` \* `num_part_each_proc`.
### 2. Run full-batch graphsage training
1. partition the graph with the instruction provided in subsection *graph partition*.
2. back to top directory of this project, then `cd examples/graphsage/`
3. change the input data dir: by modifying the yaml files in `config/fugaku/`. the input_dir must to be set as the father directory of `out_dir` indicated in the previous (postprocess) step . 
4. also you can change the model hyperparameters by modifying the yaml files.
5. back to the `examples/graphsage/`, then use following command to run the full-batch graphsage training:
	1. `mpirun -np ${num_procs} python train.py --config=./config/fugaku/${graph_name}.yaml`
		- `-np (num_procs)`: number of MPI processes for training
		- `--config`: the training config file located at `config/fugaku/`
		- `--num_bits`: number of bits for boundary node communication, option: \[32, 16, 8, 4, 2\], default: `32`
		- `--is_pre_delay`: use pre-post aggregation for communication, option: \[true, false\], default: `false`
		- `--is_label_augment`: use label augmentation, option: \[true, false\], default: `false`
	2. Example for running full-batch graphsage training on ogbn-product dataset using 32 mpi processes, int2 for communication, enable pre-post aggregation for communication, and label augmentation: `mpirun -np 32 python train.py --config=./config/fugaku/ogbn-products.yaml --num_bits=2 --is_pre_delay=true --is_label_augment=true` 
### 3. Reproduce our experiment result on Fugaku
1. partition the graph with the instruction provided in subsection *graph partition*.
2. change the input data dir: by modifying the yaml files in `config/fugaku/`. the input_dir must to be set as the father directory of `out_dir` indicated in the previous (postprocess) step. 
3. back to the top directory of this project. then `bash submit_scripts_at_fugaku/${graph_name}/submit_scripts/submit_${graph_name}_all.sh`. replace the \${graph_name} with the graph name. option: \[ogbn-arxiv, ogbn-products, reddit, proteins, ogbn-papers100M, ogbn-mag240M_paper_cites_paper\]
### 4. Reproduce our experiment result on ABCI
Since we update the code a lot, the result of this repo might not match with the result in our paper. To reproduce the result of ABCI in our paper, please refer to the original repo: [empty link](xxx)
1. partition the graph with the instruction provided in subsection *graph partition*.
2. change the input data dir: by modifying the yaml files in `config/abci/`. the input_dir must to be set as the father directory of `out_dir` indicated in the previous (postprocess) step. 
3. back to the top directory of this project. then `bash submit_scripts_at_abci/batch_job_submission/submit_${graph_name}_all.sh`. replace the \${graph_name} with the graph name. option: \[ogbn-arxiv, ogbn-products, reddit, proteins, ogbn-papers100M, ogbn-mag240M_paper_cites_paper\] 
