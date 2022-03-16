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

from contextlib import ContextDecorator
from collections import defaultdict
import copy
import threading

class OptPipelines:

    pipelines = defaultdict(None)

    @classmethod
    def register_pipeline(cls, name, func):
        assert name not in cls.pipelines, f"The pipeline {name} had already registered"
        cls.pipelines[name] = func


def _check_dynamic_ranges(val):
    assert isinstance(val, dict), "Dynamic ranges of trt should be a dict."
    assert "min" in val and "max" in val and "opts" in val, "min/max/opt should be set for dynamic ranges."
    min_shape = val["min"]
    max_shape = val["max"]
    opt_shapes = val["opts"]

    inp_nums = len(min_shape)
    assert inp_nums == len(max_shape), "the number of inputs should be equal between min_shape and max_shape"
    for min_inp, max_inp in zip(min_shape, max_shape):
        assert all(dim_min <= dim_max for dim_min, dim_max in zip(min_inp, max_inp)), \
            "The number in min_shape needs to be less than or equal to max_shape"

    for single_opt_shape in opt_shapes:
        assert inp_nums == len(single_opt_shape), \
            "the number of inputs of each group in opts must be consistent with the min_shape and max_shape"
        for min_inp, opt_inp, max_inp in zip(min_shape, single_opt_shape, max_shape):
            assert all(dim_min <= dim_opt <= dim_max for dim_min, dim_opt, dim_max in zip(min_inp, opt_inp, max_inp)), \
                "The number in opt_shape needs to be between min_shape and max_shape"


def _validate_dynamic_ranges(val):
    if isinstance(val, dict):
        val = [val]
    for v in val:
        _check_dynamic_ranges(v)
    return val


def _validate_extra_dynamic_ranges(val):
    val = _validate_dynamic_ranges(val)
    for v in val:
        assert "extra_inputs" in v, "extra_inputs should be list in extra dynamic ranges"
        assert len(v["extra_inputs"]) == len(v["min"]), "the number of inputs should be equal to number of min/max/opt shapes"
    return val


class ConfigContext(ContextDecorator):
    context = threading.local()
    context.dict = defaultdict(list)

    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        # no race-condition here, cls.context is a thread-local object
        # be sure not to override contexts in a subclass however!
        return cls.context.dict[cls.__name__]

    @classmethod
    def get_current_context(cls):
        """Return the deepest context on the stack."""
        if len(cls.get_contexts()) > 0:
            return cls.get_contexts()[-1]
        else:
            return None

    def clone(self):
        return copy.deepcopy(self)


class Config(ConfigContext):
    """
    The configuration for torch blade |Config|

    .. |Config| replace:: :class:`.Config`

    Example::

      import blade
      import torch_blade

      config = torch_blade.Config()
      config.enable_fp16 = False
      with config:
        # do optimization under configure with mlir amp enable
        blade.optimize(module, ...)
    """

    def __init__(self):
        super().__init__()
        # ratio of ops that will be fallback to fp32 to
        # get higher accuracy performance
        self._fp16_fallback_op_ratio = 0.0
        # Allow BladeDISC to do some AMP optimization if set.
        self._enable_fp16 = False
        # Controls the extent that BladeDISC is allowed to use fast math for
        # acceleration. Higher number usually means faster speed while it may
        # lead to some accuracy loss in some cases.
        #   Level 0: no fast math
        #   Level 1: apply approximation for some expensive math ops (e.g. exp, sin)
        #   Level 2: Level 1 + AllowReassoc
        #   Level 3: Level 2 + NoNaNs + NoSignedZeros
        #   Level 4: Level 3 + fully llvm fast math
        self._disc_cpu_fast_math_level = 4
        # min/max/opt settings for tuning trt engines with dynamic input shapes
        # looks like:
        # {
        #   "min": [[1, 3, 224, 224], [1, 50]],    # lower bound of the dynamic range of each inputs
        #   "opts": [[[1, 3, 512, 512], [1, 60]]],    # shapes that should be optimized like static shapes
        #   "max": [[1, 3, 1024, 1024], [1, 70]]     # upper bound of the dynamic range
        # }
        # note that there can be multiple `opt` shapes like:
        # "opts": [
        #           [[1, 3, 512, 512], [1, 60]],
        #           [[1, 3, 320, 320], [1, 55]],
        #        ]
        # Also, you can pass multiple dynamic input shapes if single configure does not give satisfied
        # performance, like config.dynamic_tuning_shapes = [configure1, configure2, ...].
        self._dynamic_tuning_shapes = {}
        self._extra_dynamic_tuning_shapes = {}
        self._preserved_attributes = []
        self._customize_onnx_opset_version = None
        self._enable_force_to_cuda = False
        self._enable_onnx_shape_white_list = True
        self._customize_op_white_list = []
        self._customize_op_black_list = []
        self._customize_jit_passes = []
        self._opt_pipeline = 'DISC'

    @property
    def optimization_pipeline(self):
        """ The optimization pipeline to be executed

        :type: str
        :default: 'DISC'
        """
        return self._opt_pipeline

    @optimization_pipeline.setter
    def optimization_pipeline(self, val):
        assert isinstance(val, str), "optimization_pipeline should be str, got {}".format(type(val))
        assert val in OptPipelines.pipelines, f"The pipeline {val} haven't been registered"
        self._opt_pipeline = val

    @property
    def enable_onnx_shape_white_list(self):
        """The flag is used to force convert shape aten operations to TensorRT. Currently the list contains, 
        'aten::view', 'aten::size', 'aten::reshape', 'aten::floor_divide', 'aten::Int', 'prim::NumToTensor'.

        :type: bool
        :default: True
        """
        return self._enable_onnx_shape_white_list

    @enable_onnx_shape_white_list.setter
    def enable_onnx_shape_white_list(self, val):
        assert isinstance(val, bool), "enable_onnx_shape_white_list should be bool, got {}".format(type(val))
        self._enable_onnx_shape_white_list = val

    @property
    def fp16_fallback_op_ratio(self):
        """The ratio of ops that will be fallback to fp32 to get higher accuracy performance.

        :setter: Sets a value in range [0, 1].
        :type: float
        :default: 0.0
        """
        return self._fp16_fallback_op_ratio

    @fp16_fallback_op_ratio.setter
    def fp16_fallback_op_ratio(self, val):
        assert 0 <= val <= 1.0, "fp16_fallback_op_ratio should be in range [0, 1], got {}".format(val)
        self._fp16_fallback_op_ratio = val

    @property
    def enable_mlir_amp(self):
        """[Deprecated] Please use enable_fp16.
        The flag to enable mlir amp.

        :type: bool
        :default: False
        """
        return self._enable_fp16

    @enable_mlir_amp.setter
    def enable_mlir_amp(self, val):
        assert isinstance(val, bool), "enable_mlir_amp should be bool, got {}".format(type(val))
        self._enable_fp16 = val


    @property
    def enable_fp16(self):
        """The flag to enable amp(aka fp16).

        :type: bool
        :default: False
        """
        return self._enable_fp16

    @enable_fp16.setter
    def enable_fp16(self, val):
        assert isinstance(val, bool), "enable_fp16 should be bool, got {}".format(type(val))
        self._enable_fp16 = val

    @property
    def disc_cpu_fast_math_level(self):
        """The flag to enable disc fast math.

        Level 0: no fast math
        Level 1: apply approximation for some expensive math ops (e.g. exp, sin)
        Level 2: Level 1 + AllowReassoc
        Level 3: Level 2 + NoNaNs + NoSignedZeros
        Level 4: Level 3 + fully llvm fast math

        :type: int
        :default: 4
        """
        return self._disc_cpu_fast_math_level

    @disc_cpu_fast_math_level.setter
    def disc_cpu_fast_math_level(self, val):
        assert isinstance(val, int), "disc_cpu_fast_math_level should be int, got {}".format(type(val))
        self._disc_cpu_fast_math_level = val

    @property
    def dynamic_tuning_shapes(self):
        """The dynamic shapes configuration for TensorRT & TVM tuning

        :type: dict or List of dict

        Examples::

          import blade
          import torch_blade

          config = torch_blade.Config()
          # min/max/opt settings for tuning trt engines with dynamic input shapes
          # looks like:
          config.dynamic_tuning_shapes = {
            # lower bound of the dynamic range of each inputs
            "min": [[1, 3, 224, 224], [1, 50]],
            # shapes that should be optimized like static shapes
            "opts": [[[1, 3, 512, 512], [1, 60]]],
            # upper bound of the dynamic range
            "max": [[1, 3, 1024, 1024], [1, 70]]
          }
          # note that there can be multiple `opt` shapes like:
          config.dynamic_tuning_shapes["opts"] = [
            [[1, 3, 512, 512], [1, 60]],
            [[1, 3, 320, 320], [1, 55]],
          ]
          # note that there can be multiple dynamic_tuning_shapes like:
          dynamic_tuning_shapes1 = {
            "min": [[1, 3, 224, 224], [1, 50]],
            "opts": [[[1, 3, 256, 256], [1, 60]]],
            "max": [[1, 3, 512, 512], [1, 70]]
          }
          dynamic_tuning_shapes2 = {
            "min": [[1, 3, 512, 512], [1, 70]],
            "opts": [[[1, 3, 512, 512], [1, 70]]],
            "max": [[1, 3, 1024, 1024], [1, 80]]
          }
          dynamic_tuning_shapes = [dynamic_tuning_shapes1, dynamic_tuning_shapes2, ...]
          config.dynamic_tuning_shapes = dynamic_tuning_shapes
        """
        return self._dynamic_tuning_shapes

    @dynamic_tuning_shapes.setter
    def dynamic_tuning_shapes(self, val):
        val = _validate_dynamic_ranges(val)
        self._dynamic_tuning_shapes = val

    @property
    def extra_dynamic_tuning_shapes(self):
        return self._extra_dynamic_tuning_shapes

    @extra_dynamic_tuning_shapes.setter
    def extra_dynamic_tuning_shapes(self, val):
        val = _validate_extra_dynamic_ranges(val)
        self._extra_dynamic_tuning_shapes = val

    @property
    def preserved_attributes(self):
        """A list of attributes to preserve in addition to the forward method.
        Without the list all attributes would be optimized during optimization.

        Example (Optimize a module with preserved attributes)::

            import torch
            import blade
            import torch_blade

            class MyModule(torch.nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()
                    self.modified_tensor = torch.tensor(10.)
                    self.version = 1

                def forward(self, input):
                    self.modified_tensor += 1
                    return input + self.modified_tensor

            scripted_module = torch.jit.script(MyModule().eval())
            config = torch_blade.Config()
            config.preserved_attributes = ["version"]
            with config:
                blade.optimize(scripted_module, ...)
        """
        return self._preserved_attributes

    @preserved_attributes.setter
    def preserved_attributes(self, val):
        assert isinstance(val, list), "preserved_attributes should be list, got {}".format(type(val))
        self._preserved_attributes = val

    @property
    def customize_onnx_opset_version(self):
        return self._customize_onnx_opset_version

    @customize_onnx_opset_version.setter
    def customize_onnx_opset_version(self, version):
        import torch
        from torch.onnx.symbolic_helper import _default_onnx_opset_version, _onnx_stable_opsets
        TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
        if TORCH_VERSION < (1, 8):
            from torch.onnx.symbolic_helper import _onnx_master_opset
        else:
            from torch.onnx.symbolic_helper import _onnx_main_opset
            _onnx_master_opset = _onnx_main_opset
        assert version == _default_onnx_opset_version or version in _onnx_stable_opsets + [_onnx_master_opset]
        self._customize_onnx_opset_version = version

    @classmethod
    def get_current_context_or_new(cls):
        """Return the deepest context on the stack or create a new Config."""
        return cls.get_current_context() or Config()

    @property
    def enable_force_to_cuda(self):
        """The flag enable cpu to cuda device modification.

        :type: bool
        :default: False
        """
        return self._enable_force_to_cuda

    @enable_force_to_cuda.setter
    def enable_force_to_cuda(self, val):
        assert isinstance(val, bool), "enable_force_to_cuda should be bool, got {}".format(type(val))
        self._enable_force_to_cuda = val

    @property
    def customize_op_white_list(self):
        """A list of TorchScript operations kind from aten or prim namespace.

        Those operators in the customize_op_white_list would be regarded as supported during
        the optimization clustering subgraph to backends. Note that if an operation have been
        added to the black list, it would be ignored.

        Example ::

            config = torch_blade.Config()
            # You could add all nodes of a specific kind to the white list.
            config.customize_op_white_list = ['aten::flatten']
            # Also, you could add the specific node with the index of its kind to the white list.
            # For example, the following added the first and last aten::view node.
            config.customize_op_white_list = ['aten::view@0', 'aten::view@-1']
        """
        return self._customize_op_white_list

    @customize_op_white_list.setter
    def customize_op_white_list(self, val):
        assert isinstance(val, list), "customize_op_white_list should be list, got {}".format(type(val))
        self._customize_op_white_list = val

    @property
    def customize_op_black_list(self):
        """A list of TorchScript operations kind from aten or prim namespace.

        Those operators in the customize_op_black_list would be leftover during
        the optimization clustering subgraph to backends.

        Example ::

            config = torch_blade.Config()
            # You could add all nodes of a specific kind to the black list.
            config.customize_op_black_list = ['aten::flatten']
            # Also, you could add the specific node with the index of its kind to the black list.
            # For example, the following added the first and last aten::view node.
            config.customize_op_black_list = ['aten::view@0', 'aten::view@-1']
        """
        return self._customize_op_black_list

    @customize_op_black_list.setter
    def customize_op_black_list(self, val):
        assert isinstance(val, list), "customize_op_black_list should be list, got {}".format(type(val))
        self._customize_op_black_list = val

    @property
    def customize_jit_passes(self):
        """A list custom Graph rewrite passes of TorchScript

        Example ::

            def my_custom_fuse_jit_pass(graph):
                torch._C._jit_pass_custom_pattern_based_rewrite_graph('''
                graph(%a, %b, %c, %d):
                  %q = aten::mul(%a, %b)
                  %r = aten::add(%q, %c, %d)
                  return (%r)''', '''
                graph(%a, %b, %c, %d):
                  %r = my::muladd(%a, %b, %c, %d)
                  return (%r)''', graph)

            config = torch_blade.Config()
            # You could add all nodes of a specific kind to the black list.
            config.customize_jit_passes = [my_custom_fuse_jit_pass]
            with config:
                torch_blade.optimize(...)
        """
        return self._customize_jit_passes

    @customize_jit_passes.setter
    def customize_jit_passes(self, val):
        assert isinstance(val, list), "customize_jit_passes should be list, got {}".format(type(val))
        self._customize_jit_passes = val
