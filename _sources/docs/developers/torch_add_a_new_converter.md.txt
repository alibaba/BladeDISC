# How To Add a New Torch Operator Converter

## Get the Operator's Function Schema

You can find aten native node schema definitions at
[ATen/native](https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native);
The TorchBlade also provides a tool function
[`node_schema_str`](https://github.com/alibaba/BladeDISC/blob/main/pytorch_blade/src/compiler/jit/tool_funcs.cpp#L110)
that returns the schema of the input node.

```python
import torch
import torch_blade.tools as tools

@torch.jit.script
def add(x, y):
    return x + y

print(add.graph)
for n in add.graph.nodes():
    print(tools.node_schema_str(n))
```

```text
graph(%x.1 : Tensor,
      %y.1 : Tensor):
  %4 : int = prim::Constant[value=1]()
  %5 : Tensor = aten::add(%x.1, %y.1, %4)
  return (%5)


aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> (Tensor)
```

## Create an Operator Converter

The key concepts used in this step:

- `MhloConversionContext`: The context that MLIR `Module`, `Builder`, and
  `Value` mapping are stored in during a conversion
- `MhloConverterRegistery`: The global registry that all the predefined
  converters are registered to
- `OpConverter`: A function that converts torch aten operator to MHLO
- `ConversionPattern`: A pattern-based mapping between `FunctionSchema` and
  `OpConverter`

To add the support of a new operator, please write an `OpConverter` for the
torch aten operator and register it to the registery. For example:

```C++
namespace torch {
namespace blade {
bool ConvertAtenRelu(MhloConversionContext& ctx, const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& ml_input = ctx.GetMlirValue(node.input(0));
  auto builder = *ctx.builder;
  auto elem_type = mlir::mhlo::GetMlirTensorElemType(ml_input);
  auto zero = mlir::mhlo::BuildHloConstZeroForType(builder, loc, elem_type);
  const auto& relu = mlir::mhlo::BuildMlirBinaryOp<mlir::chlo::BroadcastMaxOp>(
      builder, loc, ml_input, zero, elem_type);
  ctx.value_map[node.output(0)] = relu;
  // return true because the operator "aten::relu(Tensor self) -> Tensor" is supported
  return true;
}

auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern("aten::relu(Tensor self) -> Tensor", ConvertAtenRelu);

} // namespace blade
} // namespace torch
```

It's recommended that all torch irrelevant conversion codes are written in
`mhlo_builder`. Because it is expected that `mhlo_builder` can be reused when
another frontend other than torch is introduced.

## Add a Unit Test

A unit test is also required. Please refer to the unit tests in
`pytorch_blade/tests/mlir`, an example is:

```python
class TestDiscActivation(DiscTestCase):
    def test_relu(self, activation_func):
        relu = torch.nn.ReLU()
        x = torch.randn([2, 4, 16, 16], device=self.device)
        self._test_cvt_to_disc(relu, (x,))

```
