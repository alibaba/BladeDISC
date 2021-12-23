import contextlib
import torch_blade
from torch_blade import utils
from torch_blade import pass_manager
from torch_blade.clustering.support_fusion_algorithm import group_supported_clusters
from torch_blade.logging import logger

_MIN_GROUP_NODES = 3

@contextlib.contextmanager
def min_group_nodes(min_num_nodes):
    global _MIN_GROUP_NODES
    old_num_nodes = _MIN_GROUP_NODES
    try:
        _MIN_GROUP_NODES = min_num_nodes
        yield
    finally:
        _MIN_GROUP_NODES = old_num_nodes


def _fuse_supported_subgraph(graph, nodes_to_fuse, q_info=None):
    assert(len(nodes_to_fuse) > 0)
    group = graph.createFusionGroup()  # group is a node in graph
    # subgraph is another graph binding to group node
    subgraph = group.g('Subgraph')
    reversed_topo_nodes = reversed(nodes_to_fuse)
    node0 = nodes_to_fuse[-1]

    group.insertAfter(node0)

    # merge nodes into group in reverse order
    for node in reversed_topo_nodes:
        # 1. merge node
        try:
            # pylint: disable=maybe-no-member
            node_merged = torch_blade.tools.merge_node_into_group(group, node)
        except Exception as error:
            logger.exception(error)
            raise RuntimeError(
                "%s merge failed, please report a bug" % node.kind())

        # 2. merge outputs
        outputs = node.output_list()
        new_outputs = node_merged.output_list()
        assert(len(outputs) == len(new_outputs))

        if q_info is not None:
            for out_old, out_new in zip(outputs, new_outputs):
                q_info.add_mapping(out_old, out_new, group)

        #  If any of the outputs are still used then we need to add them
        for i, out in enumerate(outputs):
            if (len(out.uses()) == 0):
                continue
            subgraph.registerOutput(new_outputs[i])
            group_out = group.addOutput()
            out.replaceAllUsesWith(group_out)
            group_out.setType(out.type())
            if q_info is not None:
                q_info.add_mapping(out, group_out, group)
        # 3. node destroy
        node.destroy()

    # check has no constant inputs/outputs
    for inp in group.inputs():
        if (inp.node().kind() == 'prim::Constant'):
            raise RuntimeError(
                "prim::Constant should not be a subgraph input, please report a bug")

    for out in subgraph.outputs():
        if (out.node().kind() == 'prim::Constant'):
            raise RuntimeError(
                "prim::Constant should not be a subgraph output, please report a bug")

    if q_info is not None:
        for old_inp, new_inp in zip(group.input_list(), subgraph.input_list()):
            q_info.add_mapping(old_inp, new_inp, group)

def supported_node_fusion(graph, block, unsupported_nodes, q_info=None, support_number_ios=False):
    # 1. clustering supported area according to support information
    fusion_groups = group_supported_clusters(block, unsupported_nodes, support_number_ios)

    # 2. filter with threshold & or some rules, and construct prim::FusionGroup
    for grp_to_fuse in fusion_groups:
        if (len(grp_to_fuse) < _MIN_GROUP_NODES):
            continue
        _fuse_supported_subgraph(graph, grp_to_fuse, q_info)

    utils.block_topology_ajust(block)

    # 3. eliminate some dead constants
    pass_manager._jit_pass_dce_during_lower_to_trt(graph)

def conv_centric_fusion(graph, block, unsupported_nodes, q_info=None):
    # TODO: check conv downstream ops to see if fusion is possible
    conv_ops = ['aten::conv2d', 'aten::_convolution']
    grps = [[grp] for grp in [x for x in block.node_list() if x.kind() in conv_ops]]
    for grp_to_fuse in grps:
        _fuse_supported_subgraph(graph, grp_to_fuse, q_info)

    utils.block_topology_ajust(block)

    # 3. eliminate some dead constants
    pass_manager._jit_pass_dce_during_lower_to_trt(graph)
