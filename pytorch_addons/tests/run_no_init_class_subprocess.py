import sys
import torch
import torch_addons  # noqa: F401

shape = [8, 3, 224, 224]
example = torch.randn(shape, device=torch.device('cuda'))
module = torch.jit.load(sys.argv[1])
print(module.forward.graph)
module(example)
