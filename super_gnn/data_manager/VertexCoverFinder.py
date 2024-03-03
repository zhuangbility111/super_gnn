# This module uses material from the Wikipedia article Hopcroft--Karp algorithm
# <https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm>, accessed on
# January 3, 2015, which is released under the Creative Commons
# Attribution-Share-Alike License 3.0
# <http://creativecommons.org/licenses/by-sa/3.0/>. That article includes
# pseudocode, which has been translated into the corresponding Python code.
#
# Portions of this module use code from David Eppstein's Python Algorithms and
# Data Structures (PADS) library, which is dedicated to the public domain (for
# proof, see <http://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt>).
"""Provides functions for computing maximum cardinality matchings and minimum
weight full matchings in a bipartite graph.

If you don't care about the particular implementation of the maximum matching
algorithm, simply use the :func:`maximum_matching`. If you do care, you can
import one of the named maximum matching algorithms directly.

For example, to find a maximum matching in the complete bipartite graph with
two vertices on the left and three vertices on the right:

>>> G = nx.complete_bipartite_graph(2, 3)
>>> left, right = nx.bipartite.sets(G)
>>> list(left)
[0, 1]
>>> list(right)
[2, 3, 4]
>>> nx.bipartite.maximum_matching(G)
{0: 2, 1: 3, 2: 0, 3: 1}

The dictionary returned by :func:`maximum_matching` includes a mapping for
vertices in both the left and right vertex sets.

Similarly, :func:`minimum_weight_full_matching` produces, for a complete
weighted bipartite graph, a matching whose cardinality is the cardinality of
the smaller of the two partitions, and for which the sum of the weights of the
edges included in the matching is minimal.

"""
import collections
import itertools

import networkx as nx
from networkx.algorithms.bipartite import sets as bipartite_sets
from networkx.algorithms.bipartite.matrix import biadjacency_matrix

import time
import multiprocessing as mp
import os

import numpy as np
import torch


INFINITY = float("inf")


def hopcroft_karp_matching(G, top_nodes=None):
    """Returns the maximum cardinality matching of the bipartite graph `G`.

    A matching is a set of edges that do not share any nodes. A maximum
    cardinality matching is a matching with the most edges possible. It
    is not always unique. Finding a matching in a bipartite graph can be
    treated as a networkx flow problem.

    The functions ``hopcroft_karp_matching`` and ``maximum_matching``
    are aliases of the same function.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container of nodes

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matches`, such that
      ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matches`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented with the `Hopcroft--Karp matching algorithm
    <https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm>`_ for
    bipartite graphs.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    maximum_matching
    hopcroft_karp_matching
    eppstein_matching

    References
    ----------
    .. [1] John E. Hopcroft and Richard M. Karp. "An n^{5 / 2} Algorithm for
       Maximum Matchings in Bipartite Graphs" In: **SIAM Journal of Computing**
       2.4 (1973), pp. 225--231. <https://doi.org/10.1137/0202019>.

    """

    # First we define some auxiliary search functions.
    #
    # If you are a human reading these auxiliary search functions, the "global"
    # variables `leftmatches`, `rightmatches`, `distances`, etc. are defined
    # below the functions, so that they are initialized close to the initial
    # invocation of the search functions.
    def breadth_first_search():
        for v in left:
            if leftmatches[v] is None:
                distances[v] = 0
                queue.append(v)
            else:
                distances[v] = INFINITY
        distances[None] = INFINITY
        while queue:
            v = queue.popleft()
            if distances[v] < distances[None]:
                for u in G[v]:
                    if distances[rightmatches[u]] is INFINITY:
                        distances[rightmatches[u]] = distances[v] + 1
                        queue.append(rightmatches[u])
        return distances[None] is not INFINITY

    def depth_first_search(v):
        if v is not None:
            for u in G[v]:
                if distances[rightmatches[u]] == distances[v] + 1:
                    if depth_first_search(rightmatches[u]):
                        rightmatches[u] = v
                        leftmatches[v] = u
                        return True
            distances[v] = INFINITY
            return False
        return True

    # Initialize the "global" variables that maintain state during the search.
    left, right = bipartite_sets(G, top_nodes)
    leftmatches = {v: None for v in left}
    rightmatches = {v: None for v in right}
    distances = {}
    queue = collections.deque()

    # Implementation note: this counter is incremented as pairs are matched but
    # it is currently not used elsewhere in the computation.
    num_matched_pairs = 0
    while breadth_first_search():
        for v in left:
            if leftmatches[v] is None:
                if depth_first_search(v):
                    num_matched_pairs += 1

    # Strip the entries matched to `None`.
    leftmatches = {k: v for k, v in leftmatches.items() if v is not None}
    rightmatches = {k: v for k, v in rightmatches.items() if v is not None}

    # At this point, the left matches and the right matches are inverses of one
    # another. In other words,
    #
    #     leftmatches == {v, k for k, v in rightmatches.items()}
    #
    # Finally, we combine both the left matches and right matches.
    return dict(itertools.chain(leftmatches.items(), rightmatches.items()))


def eppstein_matching(G, top_nodes=None):
    """Returns the maximum cardinality matching of the bipartite graph `G`.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matching`, such that
      ``matching[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matching`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented with David Eppstein's version of the algorithm
    Hopcroft--Karp algorithm (see :func:`hopcroft_karp_matching`), which
    originally appeared in the `Python Algorithms and Data Structures library
    (PADS) <http://www.ics.uci.edu/~eppstein/PADS/ABOUT-PADS.txt>`_.

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------

    hopcroft_karp_matching

    """
    # Due to its original implementation, a directed graph is needed
    # so that the two sets of bipartite nodes can be distinguished
    left, right = bipartite_sets(G, top_nodes)
    G = nx.DiGraph(G.edges(left))
    # initialize greedy matching (redundant, but faster than full search)
    matching = {}
    for u in G:
        for v in G[u]:
            if v not in matching:
                matching[v] = u
                break
    while True:
        # structure residual graph into layers
        # pred[u] gives the neighbor in the previous layer for u in U
        # preds[v] gives a list of neighbors in the previous layer for v in V
        # unmatched gives a list of unmatched vertices in final layer of V,
        # and is also used as a flag value for pred[u] when u is in the first
        # layer
        preds = {}
        unmatched = []
        pred = {u: unmatched for u in G}
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        # repeatedly extend layering structure by another pair of layers
        while layer and not unmatched:
            newLayer = {}
            for u in layer:
                for v in G[u]:
                    if v not in preds:
                        newLayer.setdefault(v, []).append(u)
            layer = []
            for v in newLayer:
                preds[v] = newLayer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        # did we finish layering without finding any alternating paths?
        if not unmatched:
            # TODO - The lines between --- were unused and were thus commented
            # out. This whole commented chunk should be reviewed to determine
            # whether it should be built upon or completely removed.
            # ---
            # unlayered = {}
            # for u in G:
            #     # TODO Why is extra inner loop necessary?
            #     for v in G[u]:
            #         if v not in preds:
            #             unlayered[v] = None
            # ---
            # TODO Originally, this function returned a three-tuple:
            #
            #     return (matching, list(pred), list(unlayered))
            #
            # For some reason, the documentation for this function
            # indicated that the second and third elements of the returned
            # three-tuple would be the vertices in the left and right vertex
            # sets, respectively, that are also in the maximum independent set.
            # However, what I think the author meant was that the second
            # element is the list of vertices that were unmatched and the third
            # element was the list of vertices that were matched. Since that
            # seems to be the case, they don't really need to be returned,
            # since that information can be inferred from the matching
            # dictionary.

            # All the matched nodes must be a key in the dictionary
            for key in matching.copy():
                matching[matching[key]] = key
            return matching

        # recursively search backward through layers to find alternating paths
        # recursion returns true if found path, false otherwise
        def recurse(v):
            if v in preds:
                L = preds.pop(v)
                for u in L:
                    if u in pred:
                        pu = pred.pop(u)
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return True
            return False

        for v in unmatched:
            recurse(v)


def _is_connected_by_alternating_path(G, v, matched_edges, unmatched_edges, targets):
    """Returns True if and only if the vertex `v` is connected to one of
    the target vertices by an alternating path in `G`.

    An *alternating path* is a path in which every other edge is in the
    specified maximum matching (and the remaining edges in the path are not in
    the matching). An alternating path may have matched edges in the even
    positions or in the odd positions, as long as the edges alternate between
    'matched' and 'unmatched'.

    `G` is an undirected bipartite NetworkX graph.

    `v` is a vertex in `G`.

    `matched_edges` is a set of edges present in a maximum matching in `G`.

    `unmatched_edges` is a set of edges not present in a maximum
    matching in `G`.

    `targets` is a set of vertices.

    """

    def _alternating_dfs(u, along_matched=True):
        """Returns True if and only if `u` is connected to one of the
        targets by an alternating path.

        `u` is a vertex in the graph `G`.

        If `along_matched` is True, this step of the depth-first search
        will continue only through edges in the given matching. Otherwise, it
        will continue only through edges *not* in the given matching.

        """
        visited = set()
        # Follow matched edges when depth is even,
        # and follow unmatched edges when depth is odd.
        initial_depth = 0 if along_matched else 1
        stack = [(u, iter(G[u]), initial_depth)]
        while stack:
            parent, children, depth = stack[-1]
            valid_edges = matched_edges if depth % 2 else unmatched_edges
            try:
                child = next(children)
                if child not in visited:
                    if (parent, child) in valid_edges or (child, parent) in valid_edges:
                        if child in targets:
                            return True
                        visited.add(child)
                        stack.append((child, iter(G[child]), depth + 1))
            except StopIteration:
                stack.pop()
        return False

    # Check for alternating paths starting with edges in the matching, then
    # check for alternating paths starting with edges not in the
    # matching.
    return _alternating_dfs(v, along_matched=True) or _alternating_dfs(
        v, along_matched=False
    )


def process_vertex_list_part(G, matched_edges, unmatched_edges, targets, vertex_list_part, core_id):
    res = set()
    pid = os.getpid()
    # bind the process to a specific core
    # os.sched_setaffinity(pid, [core_id])
    # get the current process id and the core id
    print("The process id is: ", pid, " and the core id is: ", os.sched_getaffinity(pid), flush=True)

    count = 0
    elapsed_time = 0.0
    for v in vertex_list_part:
        begin = time.perf_counter()
        if v in targets or _is_connected_by_alternating_path(G, v, matched_edges, unmatched_edges, targets):
            res.add(v)
        end = time.perf_counter()
        count += 1
        elapsed_time += end - begin
        tid = mp.current_process().name
        if count % 100 == 0:
            print("The elapsed time for ", count, " vertices in ", tid, " is: ", elapsed_time, "s", flush=True)
            elapsed_time = 0.0
        
    return res

def _connected_by_alternating_paths(G, matching, targets):
    """Returns the set of vertices that are connected to one of the target
    vertices by an alternating path in `G` or are themselves a target.

    An *alternating path* is a path in which every other edge is in the
    specified maximum matching (and the remaining edges in the path are not in
    the matching). An alternating path may have matched edges in the even
    positions or in the odd positions, as long as the edges alternate between
    'matched' and 'unmatched'.

    `G` is an undirected bipartite NetworkX graph.

    `matching` is a dictionary representing a maximum matching in `G`, as
    returned by, for example, :func:`maximum_matching`.

    `targets` is a set of vertices.

    """
    # Get the set of matched edges and the set of unmatched edges. Only include
    # one version of each undirected edge (for example, include edge (1, 2) but
    # not edge (2, 1)). Using frozensets as an intermediary step we do not
    # require nodes to be orderable.
    edge_sets = {frozenset((u, v)) for u, v in matching.items()}
    matched_edges = {tuple(edge) for edge in edge_sets}
    unmatched_edges = {
        (u, v) for (u, v) in G.edges() if frozenset((u, v)) not in edge_sets
    }

    vertex_list = list(G.nodes())

    # print("The number of vertices in the graph is: ", len(vertex_list), flush=True)

    if len(vertex_list) <= 5:
        # return process_vertex_list_part(G, matched_edges, unmatched_edges, targets, vertex_list, 0)
        return {
            v
            for v in G
            if v in targets
            or _is_connected_by_alternating_path(
                G, v, matched_edges, unmatched_edges, targets
            )
        }
    else:
        # convert the graph to a sparse matrix with csr format, csr is stored by torch tensor
        # print(G.nodes())
        # print(G.edges())
        # top_nodes, bottom_nodes = nx.bipartite.sets(G)
        # row_order = list(top_nodes).sort()
        # column_order = list(bottom_nodes).sort()
        # adj_matrix = biadjacency_matrix(G, row_order=row_order, column_order=column_order)
        begin = time.perf_counter()
        # convert edges list to scipy sparse matrix
        adj_matrix = nx.to_scipy_sparse_array(G, format="csr")
        # convert the rowptr, colidx, and data to torch tensor
        rowptr = torch.tensor(adj_matrix.indptr, dtype=torch.int64)
        colidx = torch.tensor(adj_matrix.indices, dtype=torch.int64)
        # data = torch.tensor(adj_matrix.data, dtype=torch.int64)

        # get the G.nodes as the vertex id map list, convert it to torch tensor
        vid_local_to_global_map_tensor = torch.tensor(list(vertex_list), dtype=torch.int64)
        vid_global_to_local_map_dict = {vid_local_to_global_map_tensor[i].item(): i for i in range(vid_local_to_global_map_tensor.size(0))}
        num_nodes = vid_local_to_global_map_tensor.size(0)

        # print("vid_local_to_global_map_tensor", vid_local_to_global_map_tensor)
        # print("vid_global_to_local_map_dict", vid_global_to_local_map_dict)

        # print("rowptr", rowptr)
        # print("colidx", colidx)
        # convert the matched_edges to torch tensor, u is the index and v is the value
        num_nodes = rowptr.size(0) - 1
        # print("matched_edges", matched_edges)
        matched_edges_tensor = torch.full((num_nodes, ), -1, dtype=torch.int64)
        vertex_cover_tensor = torch.full((num_nodes, ), 0, dtype=torch.int64)

        # convert the matched_edges to torch tensor, use local vertex id
        for (u, v) in matched_edges:
            u_local = vid_global_to_local_map_dict[u]
            v_local = vid_global_to_local_map_dict[v]
            matched_edges_tensor[u_local] = v_local
            matched_edges_tensor[v_local] = u_local
        
        # convert the unmatched vertex list to torch tensor, use local vertex id
        unmatched_vertex_list = []
        for v in targets:
            unmatched_vertex_list.append(vid_global_to_local_map_dict[v])
        unmatched_vertex_tensor = torch.tensor(unmatched_vertex_list, dtype=torch.int64)
        print("unmatched_vertex_tensor.size(0)", unmatched_vertex_tensor.size(0))

        # print("unmatched_vertex_tensor", unmatched_vertex_tensor)
        end = time.perf_counter()
        print("inner time for prepaing (s): ", end - begin)
        
        begin = time.perf_counter()
        import supergnn_ops as ops
        ops.find_vertex_cover(rowptr, colidx, matched_edges_tensor, unmatched_vertex_tensor, vertex_cover_tensor)
        end = time.perf_counter()
        print("inner time for finding (s): ", end - begin)

        # convert the tensor to set
        vertex_cover_set = set()
        # if vertex_cover_tensor == 1, then add it to the vertex_cover_set
        for i in range(vertex_cover_tensor.size(0)):
            if vertex_cover_tensor[i] == 1:
                vertex_cover_set.add(vid_local_to_global_map_tensor[i].item())
        # print("vertex_cover_set", vertex_cover_set)
        return vertex_cover_set

    # num_processes = 1
    # if len(vertex_list) >= 100:
    #     num_processes = 10

    # # split the vertex list into num_processes parts
    # vertex_list_parts = []
    # results = [set() for _ in range(num_processes)]
    # part_size = len(vertex_list) // num_processes
    # for i in range(num_processes):
    #     if i == num_processes - 1:
    #         vertex_list_parts.append(vertex_list[i * part_size:])
    #     else:
    #         vertex_list_parts.append(vertex_list[i * part_size: (i + 1) * part_size])

    # if num_processes == 1:
    #     return process_vertex_list_part(G, matched_edges, unmatched_edges, targets, vertex_list, 0)
    
    # pool = mp.Pool(processes=num_processes)
    # pid = 0
    # tmp = []
    # results = set()

    # for process_vertex_list in vertex_list_parts:
    #     result = pool.apply_async(process_vertex_list_part, args=(G, matched_edges, unmatched_edges, targets, process_vertex_list, pid + 12))
    #     tmp.append(result)
    #     pid += 1

    # # 关闭进程池
    # pool.close()
    # pool.join()

    # # 获取每个进程的返回值
    # for result in tmp:
    #     results.update(result.get())
    
    return results
    # return {
    #     v
    #     for v in G
    #     if v in targets
    #     or _is_connected_by_alternating_path(
    #         G, v, matched_edges, unmatched_edges, targets
    #     )
    # }


def to_vertex_cover(G, matching, top_nodes=None):
    """Returns the minimum vertex cover corresponding to the given maximum
    matching of the bipartite graph `G`.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    matching : dictionary

      A dictionary whose keys are vertices in `G` and whose values are the
      distinct neighbors comprising the maximum matching for `G`, as returned
      by, for example, :func:`maximum_matching`. The dictionary *must*
      represent the maximum matching.

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed. But if more than one solution exists an exception
      will be raised.

    Returns
    -------
    vertex_cover : :class:`set`

      The minimum vertex cover in `G`.

    Raises
    ------
    AmbiguousSolution
      Raised if the input bipartite graph is disconnected and no container
      with all nodes in one bipartite set is provided. When determining
      the nodes in each bipartite set more than one valid solution is
      possible if the input graph is disconnected.

    Notes
    -----
    This function is implemented using the procedure guaranteed by `Konig's
    theorem
    <https://en.wikipedia.org/wiki/K%C3%B6nig%27s_theorem_%28graph_theory%29>`_,
    which proves an equivalence between a maximum matching and a minimum vertex
    cover in bipartite graphs.

    Since a minimum vertex cover is the complement of a maximum independent set
    for any graph, one can compute the maximum independent set of a bipartite
    graph this way:

    >>> G = nx.complete_bipartite_graph(2, 3)
    >>> matching = nx.bipartite.maximum_matching(G)
    >>> vertex_cover = nx.bipartite.to_vertex_cover(G, matching)
    >>> independent_set = set(G) - vertex_cover
    >>> print(list(independent_set))
    [2, 3, 4]

    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    """
    # This is a Python implementation of the algorithm described at
    # <https://en.wikipedia.org/wiki/K%C3%B6nig%27s_theorem_%28graph_theory%29#Proof>.
    L, R = bipartite_sets(G, top_nodes)
    # Let U be the set of unmatched vertices in the left vertex set.
    unmatched_vertices = set(G) - set(matching)
    U = unmatched_vertices & L
    # Let Z be the set of vertices that are either in U or are connected to U
    # by alternating paths.
    # Z = _connected_by_alternating_paths(G, matching, U)
    # # At this point, every edge either has a right endpoint in Z or a left
    # # endpoint not in Z. This gives us the vertex cover.
    # return (L - Z) | (R & Z)
    
    M = matching
    visited = dict()
    for node in G.nodes():
        visited[node] = 0
    def bfs_alternating_paths(v, Z, find_match=False):
        visited[v] = 1
        Z.add(v)
        
        if find_match:
            u = M[v] if v in M.keys() else None
            if u != None and visited[u] == 0:
                bfs_alternating_paths(u, Z, find_match=False)
        else:
            unmactched_neibors = [n for n in G.neighbors(v) if visited[n] == 0 and M[n] != v]
            for n in unmactched_neibors:
                bfs_alternating_paths(n, Z, find_match=True)
        return

    Z = set()
    for u in U:
        bfs_alternating_paths(u, Z, find_match=False)

    return (L - Z) | (R & Z)


#: Returns the maximum cardinality matching in the given bipartite graph.
#:
#: This function is simply an alias for :func:`hopcroft_karp_matching`.
maximum_matching = hopcroft_karp_matching

class VertexCoverFinder(object):
    def __init__(self, vertex_cover_file_list):
        # npy file for saving the vertex cover set
        self.vertex_cover_file_list = vertex_cover_file_list
        VertexCoverFinder.ctx = self
    
    def construct_graph(self, edges_list):
        graph = nx.DiGraph()
        graph.add_edges_from(edges_list)
        # print(graph.edges())
        return graph
    
    def find_minimum_vertex_cover(self, edges_list, dst_rank):
        vertex_cover_set = set()
        # check if the vertex cover file exists
        vertex_cover_file = self.vertex_cover_file_list[dst_rank]
        if os.path.exists(vertex_cover_file):
            print("The file {} exists.".format(vertex_cover_file), flush=True)
            vertex_cover_array = np.load(vertex_cover_file)
            # convert the numpy array to set
            for node in vertex_cover_array:
                vertex_cover_set.add(node)
            # load the result from file and return
            return vertex_cover_set

        # construct the directed graph
        graph = self.construct_graph(edges_list)

        # convert the directed graph to undirected graph
        undirected_graph = graph.to_undirected()

        begin = time.perf_counter()
        # get all connected components in the undirected graph
        connected_components = list(nx.connected_components(undirected_graph))
        end = time.perf_counter()
        print("Time for connected components: ", end - begin, "s", flush=True)

        # find the minimum vertex cover in each connected component
        for component in connected_components:
            subgraph = undirected_graph.subgraph(component)
            # print("The subgraph is: ", subgraph.edges())
            if nx.is_bipartite(subgraph):
                begin = time.perf_counter()
                # get the maximum matching (minimum vertex cover) in current bipartite graph
                # matching = nx.bipartite.maximum_matching(subgraph)
                matching = maximum_matching(subgraph)
                end = time.perf_counter()
                print("Time for maximum matching: ", end - begin, "s", flush=True)
                begin = time.perf_counter()
                # vertex_cover = nx.algorithms.bipartite.to_vertex_cover(subgraph, matching)
                tmp_vertex_cover = to_vertex_cover(subgraph, matching)
                end = time.perf_counter()
                print("Time for vertex cover: ", end - begin, "s", flush=True)
            else:
                print("Error: the subgraph is not bipartite")
            
            # update the set of all vertex cover with the current vertex cover
            vertex_cover_set.update(tmp_vertex_cover)
        
        # convert the set to numpy array
        vertex_cover_array = np.array(list(vertex_cover_set), dtype=np.int64)
        # save the vertex cover set to file
        np.save(vertex_cover_file, vertex_cover_array)
        
        return vertex_cover_set
    
