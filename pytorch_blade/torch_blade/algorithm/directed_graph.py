import heapq
import networkx as nx
import networkx.algorithms.shortest_paths.generic as nx_path
import networkx.algorithms.dag as nx_dag


class NxGraph(object):
    """ A wrapper Graph to networkx Graph that meets our usages """

    def __init__(self, num_nodes=0):
        self._graph = nx.DiGraph()
        for k in range(num_nodes):
            self.add_node(k)

    def add_node(self, node: int):
        self._graph.add_node(node)

    def merge_node(self, u: int, v: int):
        if (u == v):
            return
        inp_edges = [(w, u) for w, _ in self.in_edges(v) if w != u]
        out_edges = [(u, w) for _, w in self.out_edges(v) if w != u]

        new_edges = inp_edges + out_edges
        self._graph.remove_node(v)
        self._graph.add_edges_from(new_edges)

    def in_edges(self, node: int):
        return self._graph.in_edges(node)

    def out_edges(self, node: int):
        return self._graph.out_edges(node)

    def remove_node(self, node: int):
        self._graph.remove_node(node)

    def add_edge(self, src: int, dst: int):
        self._graph.add_edge(src, dst)

    def remove_edge(self, src: int, dst: int):
        self._graph.remove_edge(src, dst)

    def lexical_order_topolist(self):
        # will raise exception if the graph is not acyclic
        return list(nx_dag.lexicographical_topological_sort(self._graph))

    def has_cycle(self):
        is_acyclic = nx_dag.is_directed_acyclic_graph(self._graph)
        return not is_acyclic

    def has_path(self, src: int, dst: int):
        return nx_path.has_path(self._graph, src, dst)

    def clear(self):
        self._graph.clear()


class AdjGraph(object):

    def __init__(self, num_nodes: int):
        self._num_nodes = num_nodes
        self._adj_table = dict()

    def add_edge(self, src: int, dst: int):
        assert(src < self._num_nodes)
        assert(dst < self._num_nodes)

        if (src not in self._adj_table):
            self._adj_table[src] = set()

        self._adj_table[src].add(dst)

    def _get_in_degree(self):
        in_degree = [0] * self._num_nodes
        for _, out_set in self._adj_table.items():
            for out in out_set:
                in_degree[out] += 1

        return in_degree

    def lexical_order_topolist(self):
        input_degree = self._get_in_degree()
        heap = [idx for idx, dgr in enumerate(input_degree) if dgr == 0]
        heapq.heapify(heap)

        topo_indices = []
        while len(heap) > 0:
            # use heap as priority queue, always put node with smaller index at front
            src_idx = heapq.heappop(heap)
            topo_indices.append(src_idx)
            if (src_idx not in self._adj_table):
                continue
            succ_indices = self._adj_table[src_idx]

            for succ_idx in succ_indices:
                input_degree[succ_idx] -= 1
                if (input_degree[succ_idx] == 0):
                    heapq.heappush(heap, succ_idx)
                assert(input_degree[succ_idx] >= 0)

        return topo_indices

    def has_cycle(self):
        topo_indices = self.lexical_order_topolist()
        # if there has no cycle the topo_indices should has num_nodes elements
        if (len(topo_indices) < self._num_nodes):
            return True
        return False

    def has_path_dfs(self, x: int, y: int):
        # odd: the following param visited is stateful
        # def has_path_dfs(self, x:int, y:int, visited=set()):
        #     ...
        #     self.has_path_dfs(m, y, visited)
        #     ...
        #     return ..
        assert(x < self._num_nodes)
        assert(y < self._num_nodes)
        return self._has_path_dfs(x, y, set())

    def _has_path_dfs(self, x: int, y: int, visited):
        if (x == y):
            return True

        if (x not in self._adj_table):
            return False

        if (y in self._adj_table[x]):
            return True
        # 1. should not allow the x apear in the path twice,
        #    x --> mid --> ... -> x --> y
        #
        # 2. also if the mid is visited and there is no path:
        #    mid -> ... -> y
        #    it should not be try again
        visited.add(x)
        for mid in self._adj_table[x]:
            if (mid in visited):
                continue
            if (self._has_path_dfs(mid, y, visited)):
                return True

        return False
