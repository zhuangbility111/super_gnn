#include <torch/extension.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <list>


void find_vertex_cover(torch::Tensor rowptr, torch::Tensor colidx, 
                        torch::Tensor matched_edges_tensor, torch::Tensor unmatched_vertex_list, 
                        torch::Tensor vertex_cover_tensor) {
    
    int64_t num_nodes = rowptr.size(0) - 1;

    // get pointers to all tensors
    int64_t* rowptr_ptr = rowptr.data_ptr<int64_t>();
    int64_t* colidx_ptr = colidx.data_ptr<int64_t>();
    int64_t* matched_edges_ptr = matched_edges_tensor.data_ptr<int64_t>();
    int64_t* unmatched_vertex_ptr = unmatched_vertex_list.data_ptr<int64_t>();
    int64_t* vertex_cover_ptr = vertex_cover_tensor.data_ptr<int64_t>();

    int64_t* visited_ptr = (int64_t*)malloc(num_nodes * sizeof(int64_t));
    memset(visited_ptr, 0, num_nodes * sizeof(int64_t));

    float elapsed_time = 0.0;

    // for loop in unmatched_vertex_list
    // #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < unmatched_vertex_list.size(0); i++) {
        // insert a timer here
        // auto start = std::chrono::high_resolution_clock::now();
        // select a vertex from unmatched_vertex_list (i)
        int64_t vertex = unmatched_vertex_ptr[i];
        if (vertex_cover_ptr[vertex] == 1)
            continue;

        // bfs in the neighborhood of the vertex, need to remember the level of bfs
        std::list<int64_t> queue;
        queue.push_back(vertex);
        int64_t level = 0;
        int64_t queue_size = 1;
        int64_t next_queue_size = 0;
        while (queue.size() > 0) {
            int64_t current_vertex = queue.front();
            queue.pop_front();
            queue_size -= 1;
            if (visited_ptr[current_vertex] == 1)
                continue;
            visited_ptr[current_vertex] = 1;
            vertex_cover_ptr[current_vertex] = 1;
            if (level % 2 == 0) {
                // if the level is even, we are looking at the unmatched edges
                for (int64_t j = rowptr_ptr[current_vertex]; j < rowptr_ptr[current_vertex + 1]; j++) {
                    int64_t neighbor = colidx_ptr[j];
                    // printf("current_vertex: %d, unmatched neighbor: %d\n", current_vertex, neighbor);
                    if (visited_ptr[neighbor] == 0 && matched_edges_ptr[current_vertex] != neighbor) {
                        queue.push_back(neighbor);
                        next_queue_size += 1;
                    }
                }
            } else {
                // if the level is odd, we are looking at the matched edges
                // since each vertex has at most one matched edge, we can directly use matched_edges_ptr[i]
                int64_t matched_neighbor = matched_edges_ptr[current_vertex];
                if (matched_neighbor != -1 && visited_ptr[matched_neighbor] == 0) {
                    // printf("current_vertex: %d, matched neighbor: %d\n", current_vertex, matched_neighbor);
                    queue.push_back(matched_neighbor);
                    next_queue_size += 1;
                }
            }
            if (queue_size == 0) {
                level += 1;
                queue_size = next_queue_size;
                next_queue_size = 0;
            }
        }

		/*
        // insert a timer here
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_time += std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        if (i % 1000 == 0) {
            printf("Elapsed time for %ld-th 1000 vertex: %f sec\n", i, elapsed_time);
            elapsed_time = 0.0;
        }
		*/
    }

    free(visited_ptr);
}
