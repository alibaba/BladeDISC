from contextlib import contextmanager

import torch

from torch_blade.pass_manager import _optimize_common
from torch_blade.quantization.utils.low_precision_handler import low_precision_ops

_supported_mode = ["fp16"]


@contextmanager
def _operate_observer_tag(observers):
    for ob in observers:
        ob.on = True
    try:
        yield
    finally:
        for ob in observers:
            ob.on = False


def add_low_precision_observer(optimized_c_module, mode):
    all_observers = {}
    count = 0
    graph = optimized_c_module.forward.graph
    for node in graph.node_list():
        node_kind = node.kind()
        if node_kind in low_precision_ops:
            name = "node{}".format(count)
            node_handler = low_precision_ops[node_kind]
            node_observers = node_handler(optimized_c_module, node, name, mode)
            all_observers[node] = node_observers
            count += 1

    return all_observers


def _cal_diff(origin_output, new_output):
    total_loss = 0
    for old_oup, new_oup in zip(origin_output, new_output):
        if isinstance(old_oup, torch.Tensor) and isinstance(new_oup, torch.Tensor):
            total_loss += torch.nn.functional.mse_loss(old_oup, new_oup).item()

    return total_loss


def _get_target_key(graph):
    target_key = []
    for node in graph.node_list():
        if node.kind() in low_precision_ops:
            for oup in node.output_list():
                target_key.append(oup.debugName())
    return target_key


def cal_low_precision_sensitivity(model, example_inputs, mode="fp16"):
    assert mode in _supported_mode, "only mode: {} is supported, got {}".format(_supported_mode, mode)

    optimized_c_module = _optimize_common(model._c)
    origin_graph = optimized_c_module.forward.graph
    origin_keys = _get_target_key(origin_graph)

    all_observers = add_low_precision_observer(optimized_c_module, mode)
    new_graph = optimized_c_module.forward.graph
    new_keys = _get_target_key(new_graph)

    assert sorted(origin_keys) == sorted(new_keys), "Output debugNames have been changed."

    # since the graph is modified, forward method needs to be re-generated
    optimized_c_module.create_method_from_graph("low_precision_forward", new_graph)
    all_diff = {}
    for node, observers in all_observers.items():
        origin_output = model(*example_inputs)
        with _operate_observer_tag(observers):
            low_precision_outputs = optimized_c_module.low_precision_forward(*example_inputs)
        diff = _cal_diff(origin_output, low_precision_outputs)
        # graph used to calculate sensitivity is modified, so using node as the
        # key is useless since the node does not match the node in the origin graph.
        node_out = node.output_list()
        assert len(node_out) == 1, "number of amp node output should be 1."
        all_diff[node_out[0].debugName()] = diff

    return all_diff
