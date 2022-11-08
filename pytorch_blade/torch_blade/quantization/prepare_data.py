# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import os
import tempfile
from contextlib import contextmanager
from typing import List

import torch
import torch_blade
from torch_blade import utils
from torch_blade.clustering.support_group_conversion import (
    group_nodes, replace_group_with_engine)
from torch_blade.config import Config
from torch_blade.logging import logger
from torch_blade.tools.shape_inference import jit_add_observer


class DataCollectObserver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use a dummy tensor to make torchscript happy:)
        to_make_data_init = [torch.randn(1), ]
        self.data: List[List[torch.Tensor]] = [to_make_data_init]

    @torch.jit.export
    def collect(self, inputs: List[torch.Tensor]):
        # move the tensor to cpu to avoid cuda oom
        inputs_cpu = [inp.cpu() for inp in inputs]
        self.data.append(inputs_cpu)
        return inputs


class DataPreparer:
    """
        This class is used to collect the calibration data for each fusion group.
        Basically, the whole process will be divided into the following steps:
        1. Make each fusion group executable, thus we can do the inference on
        the graph with node fused.
        2. Collect the data using the CollectDataObserver during the inference.
        3. Save the data to a disk file and the file path is returned.

        All the three steps are achieved by modifying the graph and the detail
        would be described in each function.
    """
    executable_fusion_prefix = "libtorch_executable_fusion_group"
    collect_observer_prefix = "collect_observer"

    def __init__(self, c_module, calibration_data):
        self.c_module = c_module
        self.calibration_data = calibration_data
        # must copy the graph to avoid the original fused graph being modified
        self.graph = c_module.forward.graph.copy()

        # Create a owner to store all custom modules during the process
        self.custom_module_owner = torch.jit.ScriptModule()
        self.custom_module_owner_inp = self.graph.addInput()
        self.custom_module_owner_inp.setType(self.custom_module_owner._c._type())

        self.get_attr_nodes_for_fusion_group = []
        self.all_collect_observer_name = []
        self.all_group_name = []

    def make_fusion_group_executable(self):
        """
            This function is used to make the fusion group executable. Before this function,
            the graph is like:
                graph(%self.1 : __torch__.___torch_mangle_2.MyModel,
                      %x : Tensor,
                      %y : Tensor,
                      %5 : __torch__.torch.jit._script.ScriptModule):
                    %2 : Tensor, %3 : Tensor = prim::FusionGroup_0(%x, %y)
                    return (%2, %3)
                with prim::FusionGroup_0 = graph(%x_ : Tensor, %y_ : Tensor):
                    ...
                    ...
                    some nodes of the fusion group

            After this function, the graph is like:
                graph(%self.1 : __torch__.___torch_mangle_2.MyModel,
                      %x : Tensor,
                      %y : Tensor,
                      %5 : __torch__.torch.jit._script.ScriptModule):
                    %6 : __torch__.libtorch_executable_fusion_group_0 = prim::GetAttr[name="libtorch_executable_fusion_group_0"](%5) # noqa
                    %7 : Tensor[] = prim::CallMethod[name="forward"](%6, %x, %y)
                    %9 : Tensor, %10 : Tensor = prim::ListUnpack(%7)
                    return (%9, %10)
        """
        get_attr_nodes_for_fusion_group = []

        for idx, g_node in enumerate(group_nodes(self.graph)):
            subgraph = g_node.g("Subgraph").copy()
            group_name = f"{self.executable_fusion_prefix}_{idx}"
            self.all_group_name.append(group_name)
            fallback_module = utils.subgraph_to_module(subgraph, group_name)

            self.custom_module_owner._c._register_attribute(
                group_name, fallback_module._type(), fallback_module
            )
            attr_node = replace_group_with_engine(
                self.graph,
                self.custom_module_owner_inp,
                g_node,
                group_name,
                fallback_module._type(),
                group_inputs=False,
                engine_method_name="forward",
            )
            get_attr_nodes_for_fusion_group.append(attr_node)

        self.get_attr_nodes_for_fusion_group = get_attr_nodes_for_fusion_group

    def insert_observer(self):
        """
           Insert the DataCollectObserver to the graph. For convenience, we
           group all inputs of each fusion group into a tensor list. After this function,
           the graph is like:
                graph(%self.1 : __torch__.___torch_mangle_2.MyModel,
                      %x : Tensor,
                      %y : Tensor,
                      %5 : __torch__.torch.jit._script.ScriptModule):
                    %6 : __torch__.libtorch_executable_fusion_group_0 = prim::GetAttr[name="libtorch_executable_fusion_group_0"](%5) # noqa

                    %11 : Tensor[] = prim::ListConstruct(%x, %y)  <-- group all inputs
                    %12 : DataCollectObserver = prim::GetAttr[name="collect_observer_0"](%5)  <-- get the observer
                    %13 : Tensor[] = prim::CallMethod[name="forward"](%12, %11)  <-- call the observer
                    %14 : Tensor, %15 : Tensor = prim::ListUnpack(%13)  <-- unpack the output of the observer

                    %7 : Tensor[] = prim::CallMethod[name="forward"](%6, %14, %15)
                    %9 : Tensor, %10 : Tensor = prim::ListUnpack(%7)
                    return (%9, %10)
        """
        all_collect_observer_name = []
        for idx, get_attr in enumerate(self.get_attr_nodes_for_fusion_group):
            users = [u.user for u in get_attr.output().uses()]
            if len(users) != 1:
                raise RuntimeError("The output of prim::GetAttr of each executable should "
                                   "be used by only one .")

            # Combine the inputs of the Fusion Group into a list
            # and make the observer record the list
            forward_node = users[0]
            list_pack = self.graph.create('prim::ListConstruct')
            self.graph.appendNode(list_pack)
            list_pack.moveBefore(forward_node)
            for inp in forward_node.input_list()[1:]:
                list_pack.addInput(inp)
            list_pack.output().setType(torch_blade.tools.get_list_tensor_type())

            # insert the observer, three steps:
            #   1. register the observer
            #   2. create getattr node for the observer
            #   3. create call method node for the observer
            collect_observer = DataCollectObserver()
            collect_observer = torch.jit.script(collect_observer)
            observer_name = f"{self.collect_observer_prefix}_{idx}"
            all_collect_observer_name.append(observer_name)
            self.custom_module_owner._c._register_attribute(
                observer_name,
                collect_observer._c._type(),
                collect_observer._c
            )

            get_attr, call_method = jit_add_observer(
                self.graph,
                (self.custom_module_owner_inp, observer_name, collect_observer._c._type()),
                list_pack.output(),
                method_name="collect",
            )

            # unpack the output of the forward method
            list_unpack = self.graph.create('prim::ListUnpack')
            self.graph.appendNode(list_unpack)
            list_unpack.moveAfter(call_method)
            list_unpack.addInput(call_method.output())
            list_unpack.eraseOutput(0)
            for inp in list_pack.input_list():
                lu_output = list_unpack.addOutput()
                lu_output.setType(inp.type())
                inp.replaceAllUsesAfterNodeWith(list_unpack, lu_output)
        self.all_collect_observer_name = all_collect_observer_name

    def get_calib_data_for_each_group(self):
        all_data = []
        self.c_module.create_method_from_graph("_inference", self.graph)
        for data in self.calibration_data:
            self.c_module._inference(*data, self.custom_module_owner)

        for observer_name in self.all_collect_observer_name:
            data = self.custom_module_owner.__getattr__(observer_name).data
            # the first one is dummy data
            all_data.append(data[1:])

        # Must remove the created method or there will be a core dump
        # when use torch.jit.save to serialize the optimized model.
        self.c_module.unsafe_remove_method("_inference")

        return all_data

    def prepare(self):
        self.make_fusion_group_executable()
        self.insert_observer()
        all_data = self.get_calib_data_for_each_group()
        self.clear_attribute()
        return all_data

    def clear_attribute(self):
        for name in itertools.chain(self.all_group_name, self.all_collect_observer_name):
            self.custom_module_owner._c.unsafe_remove_type_attribute(name)


@contextmanager
def get_calib_file_for_each_group(c_module):
    calib_data_for_all_fusion_group = None
    trt_calibration_data = Config.get_current_context_or_new().quantization_calibration_data
    if trt_calibration_data is not None:
        try:
            data_preparer = DataPreparer(c_module, trt_calibration_data)
            calib_data_for_all_fusion_group = data_preparer.prepare()
        except Exception as e:
            logger.warning(f"Unable to get calib file for each graph group due to {e}.")

    with tempfile.TemporaryDirectory(prefix=".torch_blade") as tmp_dirname:
        if calib_data_for_all_fusion_group is not None:
            calib_file_path = os.path.join(tmp_dirname, "calibration_data.pt")
            torch.save(calib_data_for_all_fusion_group, calib_file_path)
            yield calib_file_path
        else:
            yield None
