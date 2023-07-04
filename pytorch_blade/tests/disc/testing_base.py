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

import contextlib
import os
import random
import unittest

import torch
from torch.testing import FileCheck
from torch_blade import mlir, optimize, utils
from torch_blade.clustering import support_fusion_group
from torch_blade.clustering.support_fusion_group import min_group_nodes
from torch_blade.config import Config
from torch_blade.mlir import is_available
from torch_blade.pass_manager import _optimize_common
from torch_blade.quantization import is_available as is_quantization_available
from torch_blade.testing.common_utils import TestCase


def skipIfOnYitian():
    return unittest.skipIf(os.popen("lscpu").read().find("svebf16") != -1, "Yitian bug was not fix")

def skipIfNoDISC():
    return unittest.skipIf(not is_available(), "DISC support was not built")


def isTorchMlirEnable():
    return True


def skipIfEnableTorchMlir():
    return unittest.skipIf(isTorchMlirEnable(), "haven't supported")


def skipTorchLE(version, msg=""):
    return unittest.skipIf(
        utils.torch_version_number() <= utils.parse_version(version),
        "TODO: torch version compatible with early than {} {}".format(version, msg))


def skipTorchNE(version, msg=""):
    return unittest.skipIf(
        utils.torch_version_number() != utils.parse_version(version),
        "TODO: torch version compatible with early than {} {}".format(version, msg))


def skipTorchLT(version, msg=""):
    return unittest.skipIf(
        utils.torch_version_number() < utils.parse_version(version),
        "TODO: torch version compatible with early than {} {}".format(version, msg))


def skipTorchGE(version, msg=""):
    return unittest.skipIf(
        utils.torch_version_number() >= utils.parse_version(version),
        "TODO: torch version compatible with greater than {} {}".format(version, msg))


@skipIfNoDISC()
class DiscTestCase(TestCase):

    def _ScriptFunction2Module(self, nn_module):
        if isinstance(nn_module, torch.jit.ScriptFunction):
            _compilation_unit = torch._C.CompilationUnit()
            c_module = torch._C.ScriptModule("gen_func", _compilation_unit, True)
            c_module.create_method_from_graph("forward", nn_module.graph)
            return torch.jit._recursive.wrap_cpp_module(c_module)
        else:
            return nn_module

    def cvt_to_disc(self, nn_module, test_data, annotations=[]):
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = mlir.backend_name()
        cfg.annotate_args = annotations
        with mlir.testing_context(), support_fusion_group.min_group_nodes(1), cfg:
            nn_module = self._ScriptFunction2Module(nn_module)
            nn_module = nn_module.eval().to(self.device)
            opt_module = optimize(
                nn_module,
                allow_tracing=True,
                model_inputs=test_data
            )
        return opt_module

    def _test_cvt_to_disc(
        self, nn_module, test_data, annotations=[], rtol=1e-6, atol=1e-3, n_engines=1
    ):
        nn_module = self._ScriptFunction2Module(nn_module)
        nn_module = nn_module.eval().to(self.device)
        result = nn_module(*test_data)
        opt_module = self.cvt_to_disc(nn_module, test_data, annotations)
        output = opt_module.forward(*test_data)
        self.assertEqual(output, result, rtol=rtol, atol=atol)
        self.assertGreaterEqual(mlir.num_engines(opt_module), n_engines)
        return output, result

    def _test_torchscipte_to_mhlo(self, graph, expected_str):
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = mlir.backend_name()
        with cfg:
            _, mhlo_graph_str, _, _ = mlir.cvt_torchscript_to_mhlo(graph)
        FileCheck().run(expected_str, mhlo_graph_str)

    def _gen_test_data(self, annotation, random_seed, lower=1, upper=10):
        test_data = []
        random.seed(random_seed)
        for dims, dtype in annotation:
            dims = [random.randint(lower, upper) if d == -1 else d for d in dims]
            if dtype.is_floating_point:
                test_data.append(torch.randn(dims, dtype=dtype, device=self.device))
            else:
                test_data.append(torch.randint(lower, upper, dims, dtype=dtype, device=self.device))
        return tuple(test_data)

    def _test_disc(self, nn_module, annotation, test_data=None, rtol=1e-6, atol=1e-3, random_seed=10):
        test_data = test_data if test_data else self._gen_test_data(annotation, random_seed)
        return self._test_cvt_to_disc(nn_module, test_data, annotation, rtol=rtol, atol=atol)


@contextlib.contextmanager
def set_env(**environ):
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


@skipIfNoDISC()
class DiscPdlTestCase(TestCase):
    def setUp(self):
        super().setUp()
        p = os.path.dirname(__file__)
        if self.device == torch.device('cuda'):
            self.device_pdll_dir = os.path.join(p, "pdl/pdll_files/gpu")
        else:
            # todo: A further distinction between x86 and aarch64 may be
            # required
            self.device_pdll_dir = os.path.join(p, "pdl/pdll_files/cpu")
        self.common_pdll_dir = os.path.join(p, "pdl/pdll_files/common")

    def _test_torchscipte_to_mhlo(
            self, module, expected_str, pdll_files=None,
            pdll_dirs=None, enable_int8=False, 
            env_var = {},
    ):
        if pdll_files is not None:
            env_var["DISC_TORCH_PDL_FILES"] = pdll_files
        if pdll_dirs is not None:
            env_var["DISC_TORCH_PDLL_INCLUDE_DIRS"] = pdll_dirs

        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = mlir.backend_name()
        cfg.enable_int8 = enable_int8
        with set_env(**env_var), cfg:
            optimized_script_module = _optimize_common(module)
            graph = optimized_script_module.forward.graph
            _, mhlo_graph_str, _, _ = mlir.cvt_torchscript_to_mhlo(graph)
        FileCheck().run(expected_str, mhlo_graph_str)

    def _test_e2e(
            self, model, inp, pdll_files=None,
            pdll_dirs=None, enable_int8=False, **kwargs
    ):
        origin_output = model(inp)
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = mlir.backend_name()
        cfg.enable_int8 = enable_int8
        env_var = {}
        if pdll_files is not None:
            env_var["DISC_TORCH_PDL_FILES"] = pdll_files
        if pdll_dirs is not None:
            env_var["DISC_TORCH_PDLL_INCLUDE_DIRS"] = pdll_dirs
        with set_env(**env_var), cfg:
            opt_model = optimize(model, True, inp)
        now_output = opt_model(inp)
        if "atol" in kwargs or "rtol" in kwargs:
            self.assertTrue(torch.allclose(origin_output, now_output, **kwargs))
        else:
            self.assertTrue(torch.equal(origin_output, now_output))


class DiscPdlQuantizationTestCase(DiscPdlTestCase):
    def setUp(self):
        super().setUp()
        self.is_quantization_available = is_quantization_available()
        if not is_quantization_available():
            self.skipTest("Pdl Quantization support is not built")


class CPUDiscPdlCase(DiscPdlTestCase):
    def setUp(self):
        super().setUp()
        if self.device != torch.device('cpu'):
            self.skipTest("Pdl test case only supports cpu platform")


class GPUDiscPdlCase(DiscPdlTestCase):
    def setUp(self):
        super().setUp()
        if self.device != torch.device('cuda'):
            self.skipTest("Pdl test case only supports gpu platform")


class CPUDiscPdlQuantizationTestCase(DiscPdlQuantizationTestCase):
    def setUp(self):
        super().setUp()
        if self.device != torch.device('cpu'):
            self.skipTest("Quantization pdl test case only supports cpu platform")


class GPUDiscPdlQuantizationTestCase(DiscPdlQuantizationTestCase):
    def setUp(self):
        super().setUp()
        if self.device != torch.device('cuda'):
            self.skipTest("Quantization pdl test case only supports gpu platform")

    def _test_e2e(
            self, model, inp, pdll_files=None,
            pdll_dirs=None, enable_int8=False,
            diff_scale=1.0
    ):
        origin_output = model(inp)
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = mlir.backend_name()
        cfg.enable_int8 = enable_int8
        env_var = {}
        if pdll_files is not None:
            env_var["DISC_TORCH_PDL_FILES"] = pdll_files
        if pdll_dirs is not None:
            env_var["DISC_TORCH_PDLL_INCLUDE_DIRS"] = pdll_dirs
        with set_env(**env_var), cfg, min_group_nodes(1):
            opt_model = optimize(model, True, inp)
        now_output = opt_model(inp)
        self.assertTrue(torch.allclose(now_output, origin_output, atol=1.0 * diff_scale))
