# Development Guide: Add a Custom Call Operator

In some scenarios, the BladeDISC code generator can not meet user needs,
e.g. vender library or calling a customized CUDA kernel code. BladeDISC
provides an interface to call these customized functions, which is called
**Custom Call Operator**, you can find more design details in
[Runtime Abstraction Layer Introduction](/docs/developers/runtime_abstraction_layer.md).

## Add a Custom Call Operator Step by Step

To make it easier for developers to understand, this section uses `TransposeOp`
as an example to introduce how to add a custom call operator step by step.

### Step1: Register Operator in RAL

BladeDISC provides a macro `TAO_RAL_API` to register a custom call operator. To
make the code structure clearly, please create a new `transpose_impl.cc` file
under the directory: `tao_compiler/mlir/ral/context/`:

``` c++
void ral_gpu_transpose_2d(ExecutionContext* ctx, void* stream_handle,
                          MemRefType<T, 2> input, int dim0, int dim1,
                          MemRefType<T, 2> output) {
  ...
  LaunchTranspose2DKernel<T>(stream, d_in, rows, cols, d_out);
}

TAO_RAL_API("ral_gpu_transpose_2d", "gpu",
            ral_gpu_transpose_2d<float>);
```

For the argument list, the first and second ones must be `ExecutionContext` and
`gpu_stream_handle`, which are used to get the GPU runtime context, the following arguments are
user-defined.

Please note that `ral_gpu_transpose_2d` function will be called at runtime, so we
can launch the GPU kernel in the function `LaunchTranspose2DKernel`, Usually you
should add the kernel implementation to the directory
`tao_compiler/mlir/ral/context/custom_library/` .

### Step2: Translate LMHLO Operator to Custom Call Operator

In the BladeDISC pass pipeline, a lmhlo operator is converted to a custom call
op via `DispatchOp` in `DiscLowerToLibraryCallPass`. A simple code snippet is as
the following:

``` c++
SmallVector<Value, 5> newOperands{stream_handle};
for (Value operand : op.getOperands()) {
      newOperands.push_back(operand);
}
rewriter.replaceOpWithNewOp<DispatchOp>(op,
      llvm::None,
      ctx, /*Ral Execution Context*/,
      newOperands, /*Arguments*/
      "ral_gpu_transpose_2d", /*registry op name*/
      false, /*has side effect or not*/
       on_gpu ? "gpu" : "cpu"/*gpu or not*/
  );
```

`ral_gpu_transpose_2d` is the target name registered in step1.

### Step3: Add a Unit Test

To make sure the custom call works well, you should add an operator unit test at
folder `tao_compiler/mlir/disc/tests/tensorflow_ops/transpose.cc`:

``` c++
TEST(TFTransposeOpTest, StaticShape2DF32) {
  EXPECT_TRUE(feature_test_main(
      /*mlir_file_path*/ c_ft_path + "transpose_2d_s_f32.mlir",
      /*backend_types*/
      {BackendType::kCuda},
      /*num_inputs*/ 1,
      /*num_outputs*/ 1,
      /*input_descriptors*/ {"6x7xf32_d"},
      /*output_descriptors*/ {"f32_d"}));
}
```
