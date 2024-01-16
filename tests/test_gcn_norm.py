import numpy as np
import torch
import torch.distributed as dist
import os
from super_gnn.communicator import Communicator
from super_gnn.data_manager.DataLoader import get_distributed_graph
from super_gnn.data_manager import CommBuffer
from super_gnn.graph_partition.postprocess_graph_multi_proc import divide_edges_into_local_and_remote
from torch_sparse import SparseTensor


class GraphGenerator(object):
    @staticmethod
    def generate_graph(num_nodes, num_edges, num_parts):
        edge_index = np.random.randint(0, num_nodes, size=(2, num_edges))
        # remove the repeated edges
        edge_index = np.unique(edge_index, axis=1)

        num_edges = edge_index.shape[1]

        parts = dict()
        for i in range(num_parts):
            parts[i] = [[], []]

        '''
        print("edge_index:", end=" ")
        for i in range(num_edges):
            print("({}, {})".format(edge_index[0][i], edge_index[1][i]), end=" ")
        print("")
        '''

        num_nodes_per_part = int((num_nodes + num_parts - 1) / num_parts)
        
        for i in range(num_edges):
            dst = edge_index[1][i]
            part = dst // num_nodes_per_part
            parts[part][0].append(edge_index[0][i])
            parts[part][1].append(edge_index[1][i])

        # print edges in each part
        for i in range(num_parts):
            print("part {}:".format(i), end=" ")
            for j in range(len(parts[i][0])):
                print("({}, {})".format(parts[i][0][j], parts[i][1][j]), end=" ")
            print("")
        
        return edge_index, parts, num_edges

    @staticmethod
    def process_graph(num_nodes, num_edges, num_parts, edge_index, parts, output_dir):
        # convert the parts (2d list) to 2d array
        for i in range(num_parts):
            parts[i] = np.array(parts[i]).transpose()

            num_nodes_per_part = int((num_nodes + num_parts - 1) / num_parts)
            begin_id = i * num_nodes_per_part
            end_id = (i + 1) * num_nodes_per_part - 1
            
            local_edges_list, remote_edges_list = divide_edges_into_local_and_remote(parts[i], begin_id, end_id)

            # save the local edges list and remote edges list
            np.save(os.path.join(output_dir, "p{:0>3d}-{}_local_edges.npy".format(i, "test")), local_edges_list)
            np.save(os.path.join(output_dir, "p{:0>3d}-{}_remote_edges.npy".format(i, "test")), remote_edges_list)

        # save the global edges list
        np.save(os.path.join(output_dir, "global_edges.npy"), edge_index)

def load_graph(input_dir, graph_name, rank, num_nodes, num_parts):
    # ----------------------------------------------------------
    # divide the global edges list into the local edges list and the remote edges list
    local_edges_list = np.load(
        os.path.join(input_dir, "p{:0>3d}-{}_local_edges.npy".format(rank, graph_name))
    )
    remote_edges_list = np.load(
        os.path.join(input_dir, "p{:0>3d}-{}_remote_edges.npy".format(rank, graph_name))
    )

    global_edges_list = np.load(os.path.join(input_dir, "global_edges.npy"))

    num_nodes_per_part = int((num_nodes + num_parts - 1) / num_parts)
    begin_id = rank * num_nodes_per_part
    end_id = (rank + 1) * num_nodes_per_part - 1
    
    if end_id >= num_nodes:
        end_id = num_nodes - 1

    original_nodes_list = np.arange(begin_id, end_id + 1)
    local_nodes_list = np.stack((original_nodes_list, original_nodes_list), axis=1)

    # get the remote nodes list from src in remote edges list, unique and sort
    remote_nodes_list = np.unique(remote_edges_list[0])


    nodes_range_on_each_subgraph = np.zeros(num_parts + 1, dtype=np.int64)
    for i in range(num_parts):
        nodes_range_on_each_subgraph[i] = i * num_nodes_per_part
    nodes_range_on_each_subgraph[num_parts] = num_nodes

    num_local_nodes = local_nodes_list.shape[0]

    local_edges_list = torch.from_numpy(local_edges_list).long()
    remote_edges_list = torch.from_numpy(remote_edges_list).long()
    local_nodes_list = torch.from_numpy(local_nodes_list).long()
    remote_nodes_list = torch.from_numpy(remote_nodes_list).long()
    nodes_range_on_each_subgraph = torch.from_numpy(nodes_range_on_each_subgraph).long()
    global_edges_list = torch.from_numpy(global_edges_list).long()

    # print("local_edges_list = {}".format(local_edges_list))
    # print("remote_edges_list = {}".format(remote_edges_list))

    dist_graph = get_distributed_graph(
        local_edges_list,
        remote_edges_list,
        local_nodes_list,
        nodes_range_on_each_subgraph,
        num_local_nodes,
        1, 
        num_parts, 
        32
    )

    return dist_graph, global_edges_list, local_nodes_list, remote_nodes_list

def gcn_norm_on_global_graph(global_edges_list, num_nodes):
    # convert the global edges list to a sparse tensor
    adj_t = SparseTensor(
        row=global_edges_list[1], 
        col=global_edges_list[0], 
        value=torch.ones(global_edges_list.shape[1], dtype=torch.float32),
        sparse_sizes=(num_nodes, num_nodes)
    )

    adj_t = adj_t.set_diag()

    # print("global adj_t = {}".format(adj_t.coo()))

    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    # print("global_deg = {}".format(deg))
    
    return adj_t

def gcn_normalize_on_dist_graph(dist_graph):
    # add dist_graph-loop
    dist_graph.local_adj_t = dist_graph.local_adj_t.fill_diag(1.0)

    # print("dist_graph.local_adj_t = {}".format(dist_graph.local_adj_t.coo()))
    # print("dist_graph.remote_adj_t = {}".format(dist_graph.remote_adj_t.coo()))

    # get in-degrees
    in_degrees = dist_graph.get_in_degrees()
    # print("local in_degrees = {}".format(in_degrees))
    deg_inv_sqrt = in_degrees.pow(-0.5)

    # normalization for local graph
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    dist_graph.local_adj_t = deg_inv_sqrt.view(-1, 1) * dist_graph.local_adj_t * deg_inv_sqrt.view(1, -1)

    # normalization for remote graph
    comm_splits = dist_graph.comm_splits
    comm_buf = CommBuffer(comm_splits, 1, 32)

    torch.index_select(deg_inv_sqrt, 0, dist_graph.idx_nodes_send_to_others, out=comm_buf.send_buf)
    dist.all_to_all_single(comm_buf.recv_buf, comm_buf.send_buf, comm_splits.recv_splits, comm_splits.send_splits)

    remote_deg_inv_sqrt = comm_buf.recv_buf.view(-1)
    print("remote_deg_inv_sqrt = {}".format(remote_deg_inv_sqrt))
    dist_graph.remote_adj_t = deg_inv_sqrt.view(-1, 1) * dist_graph.remote_adj_t * remote_deg_inv_sqrt.view(1, -1)

    # return dist_graph

def check_result(global_adj_t, dist_graph, local_nodes_list, remote_nodes_list):
    local_mat = dist_graph.local_adj_t.to_dense()
    remote_mat = dist_graph.remote_adj_t.to_dense()
    global_mat = global_adj_t.to_dense()

    # print(local_nodes_list)

    # print(remote_nodes_list)

    for i in range(local_mat.shape[0]):
        for j in range(local_mat.shape[1]):
            global_i = local_nodes_list[i][0]
            global_j = local_nodes_list[j][0]
            if local_mat[i][j] != global_mat[global_i][global_j]:
                print("error, local_mat[{}][{}] = {}, global_mat[{}][{}] = {}".format(i, j, local_mat[i][j], global_i, global_j, global_mat[global_i][global_j]))
    
    for i in range(remote_mat.shape[0]):
        for j in range(remote_mat.shape[1]):
            global_i = local_nodes_list[i][0]
            global_j = remote_nodes_list[j]
            if remote_mat[i][j] != global_mat[global_i][global_j]:
                print("error, remote_mat[{}][{}] = {}, global_mat[{}][{}] = {}".format(i, j, remote_mat[i][j], global_i, global_j, global_mat[global_i][global_j]))

if __name__ == "__main__":
    Communicator(32, False)
    rank, world_size = Communicator.ctx.init_dist_group()

    num_nodes = 1111
    num_edges = 100111
    num_parts = world_size

    if rank == 0:
        print("num_nodes = {}, num_edges = {}, num_parts = {}".format(num_nodes, num_edges, num_parts))
        edge_index, parts, num_edges = GraphGenerator.generate_graph(num_nodes, num_edges, num_parts)
        GraphGenerator.process_graph(num_nodes, num_edges, num_parts, edge_index, parts, "./data/test_graph_for_gcn_norm/")

    dist.barrier()

    # load the graph
    dist_graph, global_edges_list, local_nodes_list, remote_nodes_list \
        = load_graph("./data/test_graph_for_gcn_norm/", "test", rank, num_nodes, num_parts)

    # gcn_norm on global graph
    global_adj_t = gcn_norm_on_global_graph(global_edges_list, num_nodes)

    # gcn_norm on distributed graph
    # dist_graph = gcn_normalize_on_dist_graph(dist_graph)
    gcn_normalize_on_dist_graph(dist_graph)

    check_result(global_adj_t, dist_graph, local_nodes_list, remote_nodes_list)
