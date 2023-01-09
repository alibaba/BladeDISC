# Copyright 2021 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch_blade.logging import logger
from torch_blade.algorithm import UnionSet, NxGraph
from torch_blade.config import Config
from torch_blade.tools import read_bool_from_env

class NoCycleFusedGraphBuilder(object):

    def __init__(self, num_nodes: int):
        self._num_nodes = num_nodes
        self._union_set = UnionSet(num_nodes)
        self._graph_builder = NxGraph(num_nodes)

    def get_groups(self):
        return self._union_set.get_groups()

    def same_group(self, src: int, dst: int):
        return self._union_set.same_group(src, dst)

    def find(self, group: int):
        return self._union_set.find(group)

    def add_edge(self, src: int, dst: int):
        src_group = self.find(src)
        dst_group = self.find(dst)
        assert(src_group != dst_group)
        self._graph_builder.add_edge(src_group, dst_group)

    def in_edges(self, node: int):
        return self._graph_builder.in_edges(node)

    def out_edges(self, node: int):
        return self._graph_builder.out_edges(node)

    def has_path(self, src: int, dst: int):
        return self._graph_builder.has_path(self.find(src), self.find(dst))

    def has_cycle(self):
        return self._graph_builder.has_cycle()

    def num_groups(self):
        return self._union_set.num_sets()

    def group_topolist(self):
        return self._graph_builder.lexical_order_topolist()

    def fuse(self, grp_i: int, grp_j: int):
        assert(not self.has_cycle())

        grp_i = self._union_set.find(grp_i)
        grp_j = self._union_set.find(grp_j)

        # the same group
        if (grp_i == grp_j):
            return {grp_i}

        # merge grp_j into grp_i, that would remove grp_j from graph builder
        self._union_set.union(grp_i, grp_j)
        self._graph_builder.merge_node(grp_i, grp_j)
        merged = set()
        if (self.has_cycle()):
            merged = self._remove_cycles(grp_i)
        return merged.union({grp_i, grp_j})

    def _remove_cycles(self, u: int):
        next_nodes = [v for _, v in self.out_edges(u) if u != v]

        merged = set()
        while len(next_nodes) > 0:
            v = next_nodes.pop()
            if (v in merged):
                continue

            assert(v == self.find(v))
            v = self.find(v)
            u = self.find(u)
            # can be optimized by return a path, and compress it
            if (not self.has_path(v, u)):
                continue

            # has cycle, try to compress the cycle path by merge (u, v)
            nodes_to_add = [w for _, w in self.out_edges(v) if w != u]
            self._union_set.union(u, v)
            self._graph_builder.merge_node(u, v)
            next_nodes += nodes_to_add
            merged.add(v)

        assert(not self.has_cycle())
        return merged

def _is_tensor_or_const(val, support_number_inpts_outs):
    is_tensor = val.type().isSubtypeOf(torch._C.TensorType.get())
    is_const = val.node().kind() == 'prim::Constant'
    if (is_tensor or is_const):
        return True

    # TODO(gty): To support more number type, other than int
    # number would be regard as zero rank tensor
    is_number = val.type().isSubtypeOf(torch._C.IntType.get())
    if (support_number_inpts_outs and is_number):
        return True
    return False

def _create_graph_builder(non_const_topolist, support_number_inpts_outs, trt_unsupported):
    topo_nodes = non_const_topolist
    for n in topo_nodes:
        if (n.kind() == 'prim::Constant'):
            raise RuntimeError(
                "prim::Constant should not be fused but cloned, please report a bug")
    node2idx_map = dict([(node, idx) for idx, node in enumerate(topo_nodes)])
    control_nodes = []
    graph_builder = NoCycleFusedGraphBuilder(len(topo_nodes))
    # only construct a Graph for nodes in topo_nodes
    # graph.input_nodes/output_nodes are not contained
    for idx, node in enumerate(topo_nodes):
        # setup indexed_node's inputs/outputs
        input_deps = node.input_list() + node.control_deps()
        # filter input from nodes not in current block
        input_deps = [inp for inp in input_deps if inp.node() in node2idx_map]
        for inp in input_deps:
            inp_idx = node2idx_map[inp.node()]
            graph_builder.add_edge(inp_idx, idx)

        # TODO(fix): To workaround ASR model compilation.
        # Please relax the dependencies.
        for ctl_n in control_nodes:
            graph_builder.add_edge(node2idx_map[ctl_n], idx)

        if len(list(node.blocks())) > 0:
            control_nodes.append(node)

        # Only supported nodes would be fuse, so we don't worry about unsupported nodes.
        if node in trt_unsupported:
            continue

        for inp in input_deps:
            if _is_tensor_or_const(inp, support_number_inpts_outs):
                continue
            # handle the input node in the block that produce non-Tensor
            inp_idx = node2idx_map[inp.node()]
            # fuse the producer and consumer group of non-Tensor
            merged = graph_builder.fuse(inp_idx, idx)
            if len(merged) > 20:
                logger.warn("A large cycle of %d nodes was merged into a group" % len(merged))

    assert(len(graph_builder.group_topolist()) == graph_builder.num_groups())
    return graph_builder


def _build_group_support_info(topo_nodes, trt_unsupported, graph_builder):
    group_support_info = dict()
    union_groups = graph_builder.get_groups()
    grp_topolist = graph_builder.group_topolist()
    assert(len(union_groups) == len(grp_topolist))
    for idx, node in enumerate(topo_nodes):
        group = graph_builder.find(idx)
        is_node_supported = node not in trt_unsupported
        group_support_info[group] = is_node_supported and \
            group_support_info.get(group, True)

    return group_support_info


def _cluster_by_union_find(graph_builder, support_info):
    assert(not graph_builder.has_cycle())

    def can_merge(u, v):
        is_same_kind = support_info[u] == support_info[v]
        if (not is_same_kind):
            # print("is not the same kind")
            return False
        # u -> m -> v
        #  \       .^
        #   ------/
        # Because nx_graph is acyclic, it's not possible to have a path: m--> ... u --> v
        for _, m in graph_builder.out_edges(u):
            if m == v:
                continue
            if (graph_builder.has_path(m, v)):
                # from networkx.algorithms.simple_paths import all_simple_paths as all_path
                # print("m -> v has path")
                # print(list(all_path(nx_graph._graph, m, v)))
                return False
        return True

    group_topolist = graph_builder.group_topolist()
    # some graph unions may not converge in 10 iterations, provide customize setting from config
    # TODO: refine cluster policy
    cfg = Config.get_current_context_or_new()
    max_iter_count = cfg.disc_cluster_max_iter_count
    while max_iter_count > 0:
        max_iter_count -= 1
        # TODO: merge brother group nodes that not construct a cycle
        last_graph_len = len(group_topolist)
        for v in reversed(group_topolist):
            # sort make in_edges stable
            in_edges = list(graph_builder.in_edges(v))
            for u, _ in sorted(in_edges):
                if graph_builder.same_group(u, v): continue

                # to avoid cycle
                if (not can_merge(u, v)):
                    continue

                graph_builder.fuse(u, v)
                break
            assert(not graph_builder.has_cycle())

        found = True
        merge_horizontal = read_bool_from_env("TORCH_BLADE_EXPERIMENTAL_MERGE_HORIZONTAL_GROUPS", False)
        # merge non-connected groups
        while found and merge_horizontal:
            reverse_group_topolist = list(reversed(graph_builder.group_topolist()))
            found = False
            for idx, v in enumerate(reverse_group_topolist):
                for u in reverse_group_topolist[idx+1:]:
                    if graph_builder.same_group(u, v): continue
                    if graph_builder.has_path(u, v): continue

                    is_same_kind = support_info[u] == support_info[v]
                    if not is_same_kind:
                        continue

                    graph_builder.fuse(u, v)
                    found = True
                    break
                if found:
                    break
            assert(not graph_builder.has_cycle())

        group_topolist = graph_builder.group_topolist()
        cur_graph_len = len(group_topolist)
        assert(cur_graph_len <= last_graph_len)
        if (last_graph_len == cur_graph_len):
            break
    union_groups = graph_builder.get_groups()
    union_groups = [(grp, graph_builder.find(grp[0]))
                    for grp in union_groups if len(grp) > 0]
    supported_groups = [grp for grp, gid in union_groups if support_info[gid]]
    return supported_groups


def _broadcast_unsupported_set(block, trt_unsupported, support_number_inpts_outs):
    topo_nodes = block.node_list()
    node2idx_map = dict([(node, idx) for idx, node in enumerate(topo_nodes)])
    graph_builder = NxGraph(len(topo_nodes))

    for idx, node in enumerate(topo_nodes):
        # setup indexed_node's inputs/outputs
        input_deps = node.input_list() + node.control_deps()
        # filter input from nodes not in current block
        input_deps = [inp for inp in input_deps if inp.node() in node2idx_map]
        for inp in input_deps:
            inp_idx = node2idx_map[inp.node()]
            graph_builder.add_edge(inp_idx, idx)

    def _is_tensor_producer(node):
        outputs = node.output_list()
        return all(_is_tensor_or_const(out, support_number_inpts_outs) for out in outputs)

    # A BFS algorithm that broadcast un-supportiveness on bidirection DAG,
    # according to the usages of non-Tensor.
    visited = set()
    candidates = [n for n in trt_unsupported]
    while len(candidates) > 0:
        unspt_node = candidates.pop(0)
        if unspt_node not in node2idx_map: continue
        n_idx = node2idx_map[unspt_node]
        visited.add(unspt_node)
        new_unspt_nodes = []

        # 1. broadcast un-supportiveness to output nodes, if the current node is not tensor producer
        if not _is_tensor_producer(unspt_node):
            out_edges = graph_builder.out_edges(n_idx)
            out_nodes = [topo_nodes[k] for _, k in out_edges]
            new_unspt_nodes += out_nodes

        # 2. broadcast un-supportiveness to input nodes, if they are not tensor producer
        in_edges = graph_builder.in_edges(n_idx)
        in_nodes = [topo_nodes[k] for k, _ in in_edges]
        new_unspt_nodes += [n for n in in_nodes if not _is_tensor_producer(n)]

        # 3. filter visited unspported nodes
        new_unspt_nodes = [n for n in new_unspt_nodes if n not in visited]
        candidates += new_unspt_nodes
        trt_unsupported.update(new_unspt_nodes)
    return trt_unsupported


def group_supported_clusters(block, trt_unsupported, support_number_inpts_outs=False):
    for out in block.outputs():
        if _is_tensor_or_const(out, support_number_inpts_outs):
            continue
        trt_unsupported.add(out.node())

    trt_unsupported = _broadcast_unsupported_set(block, set(trt_unsupported), support_number_inpts_outs)
    topo_nodes = block.node_list()
    non_const_topolist = [
        n for n in topo_nodes if n.kind() != 'prim::Constant']

    logger.info("Fuse non-tensor type producers & consumers")
    # the non-tensor type producers & consumers will be fused into the same node during graph builder creation
    graph_builder = _create_graph_builder(non_const_topolist, support_number_inpts_outs, trt_unsupported)

    # fetch the group support info
    group_support_info = _build_group_support_info(
        non_const_topolist, trt_unsupported, graph_builder)

    logger.info("Try clustering with support information")
    # find cluster by union find, according to group_support_info
    supported_groups = _cluster_by_union_find(
        graph_builder, group_support_info)

    logger.debug("Summary of supported groups:")
    for idx, grp in enumerate(supported_groups):
        logger.debug(f"    cluster {idx}, num nodes: {len(grp)}.")

    fusion_groups = []
    for grp in supported_groups:
        nodes_to_fuse = []
        for node_idx in grp:
            node = non_const_topolist[node_idx]
            if (node in trt_unsupported):
                raise RuntimeError(
                    "The Node %s is unsupported, please report a bug" % (node.kind()))
            if (node.kind() == 'prim::Constant'):
                raise RuntimeError(
                    "prim::Constant should be fused but cloned, please report a bug")

            nodes_to_fuse.append(node)
        fusion_groups.append(nodes_to_fuse)
    logger.info("Pass clustering with support information")
    return fusion_groups
