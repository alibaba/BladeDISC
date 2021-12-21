import unittest
from torch_addons.mlir import is_available

def skipIfNoDISC():
    return unittest.skipIf(not is_available(), "DISC support was not built")
