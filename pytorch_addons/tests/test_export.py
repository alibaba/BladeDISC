import unittest
import torch
import torch.nn as nn
import torchvision

from torch_addons.testing.common_utils import TestCase
from torch_addons.export import export, match_submodules
from torch.testing import FileCheck
from torch_addons import version

class TestExport(TestCase):

    def setUp(self):
        super().setUp()
        model = torchvision.models.resnet18().to(self.device)
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        original_output = model(dummy_input)
        self.model = model
        self.dummy_input = dummy_input
        self.original_output = original_output

    def test_different_export(self):
        all_trace_model = export(self.model, True, (self.dummy_input,))
        all_script_model = export(self.model)
        partial_trace_model = export(self.model, ['layer2', 'layer3'], (self.dummy_input,))
        self.assertIsInstance(all_trace_model, torch.jit.ScriptModule)
        self.assertIsInstance(all_script_model, torch.jit.ScriptModule)
        self.assertIsInstance(partial_trace_model, torch.jit.ScriptModule)

        all_trace_output = all_trace_model(self.dummy_input)
        all_script_output = all_script_model(self.dummy_input)
        partial_trace_output = partial_trace_model(self.dummy_input)
        self.assertEqual(self.original_output, all_trace_output)
        self.assertEqual(self.original_output, all_script_output)
        self.assertEqual(self.original_output, partial_trace_output)

    @unittest.skipIf(not (version.cuda_available and torch.distributed.is_available()),
                     "torch.distributed is not available")
    def test_different_parallel_model(self):
        dp_model = torch.nn.DataParallel(self.model)
        dp_scripted_model = export(dp_model)
        dp_output = dp_scripted_model(self.dummy_input)
        self.assertEqual(self.original_output, dp_output)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1, init_method='tcp://127.0.0.1:64752')
        ddp_model = torch.nn.parallel.DistributedDataParallel(self.model)
        ddp_scripted_model = export(ddp_model)
        ddp_output = ddp_scripted_model(self.dummy_input)
        self.assertEqual(self.original_output, ddp_output)

    @unittest.skipIf(torch.version.__version__.split('+')[0] >= "1.7.1", "torch.no_grad() has been support")
    def test_partial_export(self):
        class SubModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 1)

            def forward(self, x):
                with torch.no_grad():
                    return self.fc(x)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = SubModel()
                self.fc = nn.Linear(1, 1)

            def forward(self, x):
                return self.fc(self.sub(x))

        dummy_imput = torch.randn(1, 1)
        model = Model()
        original_output = model(dummy_imput)
        all_scripted_model = export(model)
        all_scripted_output = all_scripted_model(dummy_imput)
        self.assertEqual(original_output, all_scripted_output)
        self.assertIsInstance(all_scripted_model.fc, torch.jit.RecursiveScriptModule)
        self.assertIsInstance(all_scripted_model.sub.fc, torch.jit.RecursiveScriptModule)
        self.assertNotIsInstance(all_scripted_model, torch.jit.RecursiveScriptModule)
        self.assertNotIsInstance(all_scripted_model.sub, torch.jit.RecursiveScriptModule)

        partial_traced_model = export(model, ['sub.fc'], dummy_imput)
        partial_traced_output = partial_traced_model(dummy_imput)
        self.assertEqual(original_output, partial_traced_output)
        self.assertIsInstance(partial_traced_model.fc, torch.jit.RecursiveScriptModule)
        self.assertIsInstance(partial_traced_model.sub.fc, torch.jit.TopLevelTracedModule)
        self.assertNotIsInstance(partial_traced_model, torch.jit.RecursiveScriptModule)
        self.assertNotIsInstance(partial_traced_model.sub, torch.jit.RecursiveScriptModule)

        fc_gstr = """graph(%self : __torch__.torch.nn.modules.linear.___torch_mangle_132.Linear,
                         %input.1 : Float(1:1, 1:1)):
                     # CHECK-COUNT-2: prim::GetAttr
                     %3 : Float(1:1, 1:1) = prim::GetAttr[name="weight"](%self)
                     %4 : Float(1:1) = prim::GetAttr[name="bias"](%self)
                     # CHECK-COUNT-4: prim::Constant
                     %16 : int = prim::Constant[value=2]()
                     %17 : None = prim::Constant()
                     %18 : bool = prim::Constant[value=0]()
                     %19 : int = prim::Constant[value=1]()
                     %20 : int = aten::dim(%input.1)
                     %21 : bool = aten::eq(%20, %16)
                     # CHECK-COUNT-2: prim::If
                     %22 : bool = prim::If(%21)
                     %ret : Float(1:1, 1:1) = prim::If(%22)
                     return (%ret)
                     """
        sub_fc_gstr = """graph(%self : __torch__.torch.nn.modules.linear.___torch_mangle_133.Linear,
                     %input : Float(1:1, 1:1)):
                 # CHECK-COUNT-2: prim::GetAttr
                 %10 : Float(1:1) = prim::GetAttr[name="bias"](%self)
                 %9 : Float(1:1, 1:1) = prim::GetAttr[name="weight"](%self)
                 %5 : Float(1:1, 1:1) = aten::t(%9)
                 # CHECK-COUNT-2: prim::Constant
                 %6 : int = prim::Constant[value=1]()
                 %7 : int = prim::Constant[value=1]()
                 # CHECK: aten::addmm
                 %8 : Float(1:1, 1:1) = aten::addmm(%10, %input, %5, %6, %7)
                 return (%8)
                 """
        FileCheck().run(fc_gstr, partial_traced_model.fc.graph)
        FileCheck().run(sub_fc_gstr, partial_traced_model.sub.fc.graph)

    def test_submodules_match(self):
        name_list = match_submodules(self.model)
        self.assertEqual(name_list, [])

        name_list = match_submodules(self.model, [torch.nn.Linear])
        self.assertEqual(name_list, ['fc'])

        name_list = match_submodules(self.model, (torch.nn.Linear,))
        self.assertEqual(name_list, ['fc'])

        name_list = match_submodules(self.model, {torch.nn.Linear})
        self.assertEqual(name_list, ['fc'])

    def test_partial_script_module_deepcopy(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1, 1)
                self.fc2 = torch.jit.trace(nn.Linear(1, 1), (torch.randn(1, 1),))
                self.fc3 = torch.jit.script(nn.Linear(1, 1))

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                x = self.fc3(x)
                return x

        def _record_deepcopy_id(mod):
            ids = {}
            for _name, _m in mod.named_modules():
                if '__deepcopy__' in _m.__dict__:
                    ids[_name] = id(_m.__dict__['__deepcopy__'])
            return ids

        def _test(script_model):
            script_output = script_model(dummy_input)
            self.assertEqual(origin_output, script_output)

            _new_id = _record_deepcopy_id(model)
            self.assertEqual(origin_id, _new_id)

        dummy_input = torch.randn(1, 1)
        model = Model()
        origin_output = model(dummy_input)
        origin_id = _record_deepcopy_id(model)

        all_script_model = export(model)
        _test(all_script_model)

        all_trace_model = export(model, True, (dummy_input,))
        _test(all_trace_model)

        partial_export_model = export(model, ['fc1'], (dummy_input,))
        _test(partial_export_model)


if __name__ == '__main__':
    unittest.main()
