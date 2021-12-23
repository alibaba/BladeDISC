import torch
import unittest

from torch.testing import FileCheck
from torch_blade import mlir

from tests.mlir.testing_utils import DiscTestCase


class TestConstOps(DiscTestCase):
    def test_const_tensor(self):
        @torch.jit.script
        def return_const():
            return torch.tensor([1, 2, 3, 4])

        return_const = self._ScriptFunction2Module(return_const)
        return_const = torch._C._freeze_module(return_const._c)
        graph = return_const.forward.graph
        graph.eraseInput(0)
        disc_bytes, _, _, _ = mlir.cvt_torchscript_to_mhlo(graph)
        expect_str = """
            module {
              # CHECK: func @main() -> tensor<4xi64>
              func @main() -> tensor<4xi64> attributes {tf.entry_function = {input_placements = "", inputs = "", outputs = "12"}} {
                # CHECK: mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
                %0 = "xla_hlo.constant"() {value = dense<[1, 2, 3, 4]> : tensor<4xi64>} : () -> tensor<4xi64> loc(unknown)
                "std.return"(%0) : (tensor<4xi64>) -> () loc(unknown)
              } loc(unknown) 
            } loc(unknown)
        """
        FileCheck().run(expect_str, disc_bytes)

    def test_const_scalar(self):
        @torch.jit.script
        def return_const_scalar():
            return torch.tensor(5)

        return_const = self._ScriptFunction2Module(return_const_scalar)
        graph = return_const.forward.graph
        graph.eraseInput(0)
        disc_bytes, _, _, _ = mlir.cvt_torchscript_to_mhlo(graph)
        expect_str = """
            module {
              func @main() -> tensor<i64> attributes {tf.entry_function = {input_placements = "", inputs = "", output_placements = "gpu", outputs = "3"}} {
                # CHECK: constant 5 : i64
                %1 = "std.constant"() {value = 5 : i64} : () -> i64
                # CHECK: mhlo.constant dense<5> : tensor<i64>
                %2 = "xla_hlo.constant"() {value = dense<5> : tensor<i64>} : () -> tensor<i64>
                "std.return"(%2) : (tensor<i64>) -> () loc(unknown)
              } loc(unknown)
            } loc(unknown)
        """
        FileCheck().run(expect_str, disc_bytes)


if __name__ == "__main__":
    unittest.main()
