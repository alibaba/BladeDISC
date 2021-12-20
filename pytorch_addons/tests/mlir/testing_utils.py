import torch

from torch_addons import mlir
from torch_addons import optimize
from torch_addons.config import Config
from torch_addons.clustering import support_fusion_group
from torch_addons.testing.common_utils import TestCase
from tests.mlir import skipIfNoDISC

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

    def cvt_to_disc(self, nn_module, test_data):
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = "DISC"
        with mlir.testing_context(), support_fusion_group.min_group_nodes(1), cfg:
            nn_module = self._ScriptFunction2Module(nn_module)
            nn_module = nn_module.eval().to(self.device)
            opt_module = optimize._optimize(
                nn_module,
                allow_tracing=True,
                model_inputs=test_data,
                static_shape=False,
            )
        return opt_module

    def _test_cvt_to_disc(
        self, nn_module, test_data, rtol=1e-6, atol=1e-3, n_engines=1
    ):
        nn_module = self._ScriptFunction2Module(nn_module)
        nn_module = nn_module.eval().to(self.device)
        result = nn_module(*test_data)
        opt_module = self.cvt_to_disc(nn_module, test_data)
        output = opt_module.forward(*test_data)
        self.assertEqual(output, result, rtol=rtol, atol=atol)
        self.assertGreaterEqual(mlir.num_engines(opt_module), n_engines)
        return output, result
