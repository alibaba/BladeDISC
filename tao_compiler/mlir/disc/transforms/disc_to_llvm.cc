/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/disc/IR/disc_ral_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_to_llvm_common.h"
#include "tensorflow/compiler/mlir/disc/transforms/rewriters.h"
#include "transforms/codegen_utils.h"
#include "transforms/placement_utils.h"

// This file implements the logic to convert disc ral ops to llvm dialect

namespace mlir {
namespace disc_ral {

using LLVM::GlobalOp;
using LLVM::LLVMFuncOp;
using StrT = SmallString<128>;

namespace {

using lmhlo_disc::PrintfOp;

constexpr const char* kRalDispatchFunctionName = "disc_ral_call";
constexpr const char* kRalGpuLaunch = "ral_kernel_launch";
constexpr const char* kRalCpuLaunch = "ral_kernel_launch";
constexpr const char* kMalloc = "alloc";
constexpr const char* kFree = "dealloc";

// Encodes a mlir type and appends the encoding to the string buffer `out`.
LogicalResult getTypeEncoding(MLIRContext* ctx, Type t, StrT& out) {
  Type llvm_pointer_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  Type index_type = IndexType::get(ctx);
  if (auto memref_type = t.dyn_cast<MemRefType>()) {
    out.append(
        Twine("m").concat(Twine(memref_type.getRank()).concat("d")).str());
    return getTypeEncoding(ctx, memref_type.getElementType(), out);
  } else if (auto int_type = t.dyn_cast<IntegerType>()) {
    out.append(Twine("i").concat(Twine(int_type.getWidth())).str());
  } else if (auto fp_type = t.dyn_cast<FloatType>()) {
    out.append(Twine("f").concat(Twine(fp_type.getWidth())).str());
  } else if (auto ctx_type = t.dyn_cast<RalExecutionContextType>() ||
                             t == llvm_pointer_type) {
    out.append("pvoid");
  } else if (t == llvm_pointer_pointer_type) {
    out.append("ppvoid");
  } else if (t.isIndex()) {
    // index is mapping to int64_t a.t.m. Re-visit this in case necessary.
    out.append("i64");
  } else {
    // unknown type
    return failure();
  }
  return success();
}

// Encodes a ral_dispatch op and appends the encoding to the string buffer
// `out`. The format:
//   encoding = separator.join(target_name, device, inputs_encode,
//   outputs_encode)
//
//   separator = '___'
//
//   target_name: name of the external function to dispatch.
//
//   device: user defined string (e.g. cpu or gpu)
//
//   inputs_encode = type_separator.join([type_encoding for type in
//   input_types])
//
//   outputs_encode = type_separator.join([type_encoding for type
//   in output_types])
//
//   type_separator = '_'
LogicalResult getDispatchOpSignatureEncoding(DispatchOp dispatch_op,
                                             StrT& out) {
  const char* separator = "___";
  // append signature prefix
  out.append(dispatch_op.call_target_name());
  out.append(separator);

  // encode backend (device) info
  out.append(dispatch_op.backend_config());
  out.append(separator);

  // encode input types
  Operation* op = dispatch_op.getOperation();
  for (auto& en : llvm::enumerate(op->getOperandTypes())) {
    if (en.index() != 0) out.append("_");
    if (failed(getTypeEncoding(op->getContext(), en.value(), out)))
      return failure();
  }
  out.append(separator);

  // encode output types
  for (auto& en : llvm::enumerate(op->getResultTypes())) {
    if (en.index() != 0) out.append("_");
    if (failed(getTypeEncoding(op->getContext(), en.value(), out)))
      return failure();
  }
  if (!op->getNumResults()) out.append("void");
  return success();
}

// Loads a global op at the current insertion point and returns the loaded
// value.
Value loadGlobalString(OpBuilder& builder, const Location& loc,
                       GlobalOp globalOp) {
  MLIRContext* ctx = builder.getContext();
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalOp);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, IntegerType::get(ctx, 64),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8)), globalPtr,
      ValueRange{cst0, cst0});
}

// Returns true if the globalOp has the same value as `value`.
bool checkGlobalOpContent(GlobalOp globalOp, StringRef value) {
  Optional<Attribute> optValue = globalOp.getValue();
  if (!optValue) return false;

  StringAttr attr = (*optValue).cast<StringAttr>();
  if (!attr) return false;

  return attr.getValue() == value;
}

// Creates a global const string op named `name` using the value if not exists
// and returns the Loaded value of this global op.
Value loadOrCreateGlobalString(PatternRewriter& rewriter,
                               SymbolTable& symbol_table, Operation* op,
                               StringRef name, StringRef value) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  GlobalOp globalOp = symbol_table.lookup<GlobalOp>(name);
  if (!globalOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(module.getBody());

    auto type = LLVM::LLVMArrayType::get(IntegerType::get(op->getContext(), 8),
                                         value.size());
    globalOp = rewriter.create<LLVM::GlobalOp>(
        op->getLoc(), type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
        rewriter.getStringAttr(value), /*alignment=*/0);

    // Update the symbol table
    symbol_table.insert(globalOp);

    rewriter.restoreInsertionPoint(ip);
  } else {
    assert(checkGlobalOpContent(globalOp, value));
  }

  return loadGlobalString(rewriter, op->getLoc(), globalOp);
}

// Converts a ral.dispatch_op to its llvm format.
struct DispatchOpToLLVMPattern : public ConvertOpToLLVMPattern<DispatchOp> {
  DispatchOpToLLVMPattern(LLVMTypeConverter& type_converter,
                          SymbolTable& symbol_table)
      : ConvertOpToLLVMPattern<DispatchOp>(type_converter),
        symbol_table_(symbol_table) {}

  // Returns the ral dispatch function and inserts the declaration if not found.
  LLVMFuncOp getOrInsertDispatchFunction(PatternRewriter& rewriter,
                                         Operation* op) const;

  // Packs the inputs and outputs into a type-erased pointer array.
  // For example, `int func(int)` -> `void func(void* args[]) where args =
  // {in_ptr, out_ptr}`
  Value rewriteInsOutsOfDispatchOp(DispatchOp dispatch_op, ValueRange operands,
                                   ConversionPatternRewriter& rewriter,
                                   SmallVectorImpl<Value>& resultPtrs) const;

  LogicalResult matchAndRewrite(
      DispatchOp dispatch_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;

 private:
  SymbolTable& symbol_table_;
};

// Returns the llvm function definition of ral dispatch op and creates it first
// if not exists.
LLVMFuncOp DispatchOpToLLVMPattern::getOrInsertDispatchFunction(
    PatternRewriter& rewriter, Operation* op) const {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  LLVMFuncOp func = symbol_table_.lookup<LLVMFuncOp>(kRalDispatchFunctionName);

  if (func) return func;

  // Try to insert the function since it's not found.
  OpBuilder::InsertionGuard guard(rewriter);
  OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(module.getBody());
  Type llvm_pointer_type =
      LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  func = rewriter.create<LLVMFuncOp>(
      op->getLoc(), kRalDispatchFunctionName,
      LLVM::LLVMFunctionType::get(
          getVoidType(),
          {
              llvm_pointer_type,        /* ral_context_t */
              llvm_pointer_type,        /* void* call_target_name */
              llvm_pointer_pointer_type /* void** args */
          },
          /*isVarArg=*/false));

  symbol_table_.insert(func);

  rewriter.restoreInsertionPoint(ip);

  return func;
}

// Packs the original inputs and outputs of the ral dispatch op to a uniform
// format.
//
// %struct = alloca(sizeof(struct { Parameters..., Results..., }))
// %array = alloca((NumParameters + NumResult) * sizeof(void *))
// for (i : [0, NumParameters))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   llvm.store parameters[i], %fieldPtr
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// for (i : [NumParameters, NumParameters + NumResult))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// return %array
Value DispatchOpToLLVMPattern::rewriteInsOutsOfDispatchOp(
    DispatchOp dispatch_op, ValueRange operands,
    ConversionPatternRewriter& rewriter,
    SmallVectorImpl<Value>& resultPtrs) const {
  MLIRContext* ctx = rewriter.getContext();
  Location loc = dispatch_op.getLoc();

  Type llvm_pointer_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  Type llvm_int32_type = IntegerType::get(ctx, 32);

  Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                 rewriter.getI32IntegerAttr(0));
  Value one = rewriter.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                rewriter.getI32IntegerAttr(1));

  SmallVector<Value, 4> arguments = getTypeConverter()->promoteOperands(
      loc, dispatch_op.getOperands(), operands, rewriter);
  SmallVector<Type, 4> argument_types;
  for (auto argument : arguments) argument_types.push_back(argument.getType());
  for (auto resultType : dispatch_op.getResultTypes())
    argument_types.push_back(getTypeConverter()->convertType(resultType));

  auto struct_type =
      LLVM::LLVMStructType::getNewIdentified(ctx, StringRef(), argument_types);
  Value struct_ptr = rewriter.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(struct_type), one, /*alignment=*/0);
  Value array_size = rewriter.create<LLVM::ConstantOp>(
      loc, llvm_int32_type, rewriter.getI32IntegerAttr(argument_types.size()));
  Value array_ptr = rewriter.create<LLVM::AllocaOp>(
      loc, llvm_pointer_pointer_type, array_size, /*alignment=*/0);

  for (auto en : llvm::enumerate(argument_types)) {
    Value index = rewriter.create<LLVM::ConstantOp>(
        loc, llvm_int32_type, rewriter.getI32IntegerAttr(en.index()));
    Value field_ptr = rewriter.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(en.value()), struct_ptr,
        ArrayRef<Value>{zero, index});
    if (en.index() < arguments.size()) {
      rewriter.create<LLVM::StoreOp>(loc, arguments[en.index()], field_ptr);
    } else {
      resultPtrs.push_back(field_ptr);
    }

    Value element_ptr = rewriter.create<LLVM::GEPOp>(
        loc, llvm_pointer_pointer_type, array_ptr, index);
    Value casted =
        rewriter.create<LLVM::BitcastOp>(loc, llvm_pointer_type, field_ptr);
    rewriter.create<LLVM::StoreOp>(loc, casted, element_ptr);
  }

  return array_ptr;
}

LogicalResult DispatchOpToLLVMPattern::matchAndRewrite(
    DispatchOp dispatch_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  StrT target_name;
  if (failed(getDispatchOpSignatureEncoding(dispatch_op, target_name))) {
    dispatch_op->emitError("unknown types in the dispatch op");
    return failure();
  }

  // Make sure the trailing zero is included in the constant.
  target_name.push_back('\0');

  Operation* op = dispatch_op.getOperation();
  Location loc = op->getLoc();
  SmallVector<Value, 3> callOpOperands;
  LLVMFuncOp dispatch_func = getOrInsertDispatchFunction(rewriter, op);

  SmallVector<Value, 1> resultPtrs;
  Value packedArgs = rewriteInsOutsOfDispatchOp(
      dispatch_op, adaptor.getOperands(), rewriter, resultPtrs);

  // the first argument is ral_context
  callOpOperands.push_back(adaptor.ctx());
  // the second argument is the target name
  callOpOperands.push_back(loadOrCreateGlobalString(
      rewriter, symbol_table_, op, target_name.str().drop_back(),
      target_name.str()));
  // the third argument is the args for target function
  callOpOperands.push_back(packedArgs);

  rewriter.create<LLVM::CallOp>(
      loc, llvm::None, mlir::SymbolRefAttr::get(dispatch_func), callOpOperands);

  SmallVector<Value, 1> results;
  llvm::transform(resultPtrs, std::back_inserter(results), [&](Value v) {
    return rewriter.create<LLVM::LoadOp>(loc, v);
  });

  rewriter.replaceOp(op, results);

  return success();
}

// A rewrite pattern to convert gpu.launch_func operations into corresponding
// runtime wrapper calls (modeled by ral.dispatch ops)
class ConvertLaunchFuncOpToRalCallPattern
    : public ConvertOpToLLVMPattern<gpu::LaunchFuncOp> {
 public:
  ConvertLaunchFuncOpToRalCallPattern(LLVMTypeConverter& type_converter,
                                      SymbolTable& symbol_table)
      : ConvertOpToLLVMPattern<gpu::LaunchFuncOp>(type_converter),
        symbol_table_(symbol_table) {}

 private:
  Value generateParamsArray(gpu::LaunchFuncOp launch_op, ValueRange operands,
                            OpBuilder& builder, int& num_arguments) const;
  Value generateKernelNameConstant(StringRef moduleName, StringRef name,
                                   Location loc, OpBuilder& builder) const;

  LogicalResult matchAndRewrite(
      gpu::LaunchFuncOp launch_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;

  SymbolTable& symbol_table_;
};

// Creates a struct containing all kernel parameters on the stack and returns
// an array of type-erased pointers to the fields of the struct. The array can
// then be passed to the CUDA / ROCm (HIP) kernel launch calls.
// The generated code is essentially as follows:
//
// %struct = alloca(sizeof(struct { Parameters... }))
// %array = alloca(NumParameters * sizeof(void *))
// for (i : [0, NumParameters))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   llvm.store parameters[i], %fieldPtr
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// return %array
Value ConvertLaunchFuncOpToRalCallPattern::generateParamsArray(
    gpu::LaunchFuncOp launch_op, ValueRange operands, OpBuilder& builder,
    int& num_arguments) const {
  MLIRContext* ctx = builder.getContext();
  Type llvm_pointer_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  Type llvm_int32_type = IntegerType::get(ctx, 32);

  Location loc = launch_op.getLoc();
  int num_kernel_operands = launch_op.getNumKernelOperands();
  auto arguments = getTypeConverter()->promoteOperands(
      loc, launch_op.getOperands().take_back(num_kernel_operands),
      operands.take_back(num_kernel_operands), builder);
  num_arguments = static_cast<int>(arguments.size());
  SmallVector<Type, 4> argument_types;
  argument_types.reserve(num_arguments);
  for (auto argument : arguments) argument_types.push_back(argument.getType());
  auto struct_type =
      LLVM::LLVMStructType::getNewIdentified(ctx, StringRef(), argument_types);
  Value one = builder.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                               builder.getI32IntegerAttr(1));
  Value struct_ptr = builder.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(struct_type), one, /*alignment=*/0);
  Value array_size = builder.create<LLVM::ConstantOp>(
      loc, llvm_int32_type, builder.getI32IntegerAttr(num_arguments));
  Value array_ptr = builder.create<LLVM::AllocaOp>(
      loc, llvm_pointer_pointer_type, array_size, /*alignment=*/0);
  Value zero = builder.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                builder.getI32IntegerAttr(0));
  for (auto en : llvm::enumerate(arguments)) {
    Value index = builder.create<LLVM::ConstantOp>(
        loc, llvm_int32_type, builder.getI32IntegerAttr(en.index()));
    Value field_ptr = builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(argument_types[en.index()]), struct_ptr,
        ArrayRef<Value>{zero, index});
    builder.create<LLVM::StoreOp>(loc, en.value(), field_ptr);
    Value element_ptr = builder.create<LLVM::GEPOp>(
        loc, llvm_pointer_pointer_type, array_ptr, index);
    Value casted =
        builder.create<LLVM::BitcastOp>(loc, llvm_pointer_type, field_ptr);
    builder.create<LLVM::StoreOp>(loc, casted, element_ptr);
  }
  return array_ptr;
}

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a fatbin in the `kRalGpuLaunch` attribute.
LogicalResult ConvertLaunchFuncOpToRalCallPattern::matchAndRewrite(
    gpu::LaunchFuncOp launch_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  if (!launch_op.asyncDependencies().empty() || launch_op.asyncToken()) {
    return rewriter.notifyMatchFailure(
        launch_op, "Cannot convert with async dependency or result.");
  }

  // Create an LLVM global with CUBIN extracted from the kernel annotation and
  // obtain a pointer to the first byte in it.
  auto kernel_module = SymbolTable::lookupNearestSymbolFrom<gpu::GPUModuleOp>(
      launch_op, launch_op.getKernelModuleName());
  if (!kernel_module) {
    launch_op.emitOpError() << "cannot find corresponding kernel module.";
    return failure();
  }

  Operation* op = launch_op.getOperation();
  Location loc = launch_op.getLoc();
  auto get_blob = [&](std::string name) -> Value {
    auto binary_attr = kernel_module->getAttrOfType<StringAttr>(name);
    if (!binary_attr) {
      return nullptr;
    }
    // Create a global for the module blob.
    StrT name_buffer(kernel_module.getName());
    name_buffer.append("_blob_");
    name_buffer.append(name);
    Value module_blob = loadOrCreateGlobalString(
        rewriter, symbol_table_, op, name_buffer.str(), binary_attr.getValue());
    return module_blob;
  };

  SmallVector<Value, 4> module_blobs;
  Value module_blob = get_blob(kGpuBinaryAttrName);
  if (module_blob != nullptr) {
    // JIT mode with only one module_blob needed
    module_blobs.push_back(module_blob);
  } else {
    // AOT mode with Multiple device type support
    for (auto item : c_MULTI_SM_CONFIG) {
      std::string name = std::string(kGpuBinaryAttrName) + "_" + item.first;
      module_blob = get_blob(name);
      if (module_blob != nullptr) {
        module_blobs.push_back(module_blob);
      }
    }
  }
  if (module_blobs.empty()) {
    kernel_module.emitOpError()
        << "missing " << kGpuBinaryAttrName
        << " attribute, at least one should be provided";
    return failure();
  }

  Type llvm_int32_type = IntegerType::get(rewriter.getContext(), 32);
  Type llvm_pointer_type =
      LLVM::LLVMPointerType::get(IntegerType::get(op->getContext(), 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  size_t blobs_size = module_blobs.size();
  Value module_blobs_size = rewriter.create<LLVM::ConstantOp>(
      loc, llvm_int32_type, rewriter.getI32IntegerAttr(blobs_size));
  Value module_blobs_array_ptr = rewriter.create<LLVM::AllocaOp>(
      loc, llvm_pointer_pointer_type, module_blobs_size, /*alignment=*/0);
  for (auto en : llvm::enumerate(module_blobs)) {
    Value index = rewriter.create<LLVM::ConstantOp>(
        loc, llvm_int32_type, rewriter.getI32IntegerAttr(en.index()));
    Value element_ptr = rewriter.create<LLVM::GEPOp>(
        loc, llvm_pointer_pointer_type, module_blobs_array_ptr, index);
    rewriter.create<LLVM::StoreOp>(loc, en.value(), element_ptr);
  }
  Value num_blobs = rewriter.create<LLVM::ConstantOp>(
      loc, getIndexType(), rewriter.getIntegerAttr(getIndexType(), blobs_size));

  // Make sure the trailing zero is included in the constant.
  auto kernel_name = launch_op.getKernelName().getValue();
  SmallString<128> kernel_name_buffer(kernel_name);
  kernel_name_buffer.push_back('\0');

  // Create a global for the kernel name.
  SmallString<128> kernel_name_global_name_buffer;
  auto kernel_name_global_name =
      (kernel_module.getName() + "_" + kernel_name + "_kernel_name")
          .toStringRef(kernel_name_global_name_buffer);
  Value kernel_name_global = loadOrCreateGlobalString(
      rewriter, symbol_table_, op, kernel_name_global_name,
      kernel_name_buffer.str());

  // The Ral Context is the first argument of the surrounding LLVMFunc.
  int num_arguments;
  Value context_arg =
      launch_op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
  auto kernel_params = generateParamsArray(launch_op, adaptor.getOperands(),
                                           rewriter, num_arguments);

  Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                 rewriter.getI32IntegerAttr(0));
  Value num_arg_value = rewriter.create<LLVM::ConstantOp>(
      loc, llvm_int32_type, rewriter.getI32IntegerAttr(num_arguments));
  Type pointer_type =
      LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8));
  Value stream_idx = rewriter.create<LLVM::IntToPtrOp>(loc, pointer_type, zero);
  // clang-format off
  // TODO(disc): we use the default stream a.t.m. Implement a stream assignment algo in case necessary.
  SmallVector<Value, 12> newOperands{
      module_blobs_array_ptr, /* fatbin strings */
      num_blobs, /* number of fatbin strings */
      kernel_name_global, /* name of the kernel to launch */
      adaptor.gridSizeX(), adaptor.gridSizeY(), adaptor.gridSizeZ(),
      adaptor.blockSizeX(), adaptor.blockSizeY(), adaptor.blockSizeZ(),
      zero, /* sharedMemBytes */
      stream_idx, /* gpu stream index */
      num_arg_value, /* num_args */
      kernel_params /* params for the kernel to launch */
  };
  // clang-format on

  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      launch_op, llvm::None, context_arg, newOperands, kRalGpuLaunch, false,
      "gpu");

  return success();
}

// A rewrite pattern to convert memref.alloc operations into corresponding
// runtime wrapper calls (modeled by ral.dispatch ops)
// Converting:
//   %output = memref.alloc(%0, %1) : memref<?x?xf32, "gpu">
//     to
//   "disc_ral.dispatch"(%ctx, %3) {backend_config = "gpu", call_target_name =
//   "alloc", has_side_effect = false} : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
// then convert to llvm
class ConvertMemRefAllocOpToDispatchOpPattern
    : public ConvertOpToLLVMPattern<memref::AllocOp> {
 public:
  ConvertMemRefAllocOpToDispatchOpPattern(LLVMTypeConverter& type_converter,
                                          SymbolTable& symbol_table)
      : ConvertOpToLLVMPattern<memref::AllocOp>(type_converter),
        symbol_table_(symbol_table) {}

 private:
  // TODO(disc): Remove strides computation.
  MemRefDescriptor CreateMemRefDescriptor(Location loc,
                                          ConversionPatternRewriter& rewriter,
                                          MemRefType memref_type,
                                          Value allocated_byte_ptr,
                                          ArrayRef<Value> sizes) const {
    auto memref_desc = MemRefDescriptor::undef(
        rewriter, loc, typeConverter->convertType(memref_type));

    Value allocated_type_ptr = rewriter.create<LLVM::BitcastOp>(
        loc, getElementPtrType(memref_type), allocated_byte_ptr);
    memref_desc.setAllocatedPtr(rewriter, loc, allocated_type_ptr);
    memref_desc.setAlignedPtr(rewriter, loc, allocated_type_ptr);
    memref_desc.setConstantOffset(rewriter, loc, 0);

    if (memref_type.getRank() == 0) {
      return memref_desc;
    }

    // Compute strides and populate descriptor `size` and `stride` fields.
    Value stride_carried = createIndexConstant(rewriter, loc, 1);
    for (int pos = sizes.size() - 1; pos >= 0; --pos) {
      Value size = sizes[pos];
      memref_desc.setSize(rewriter, loc, pos, size);
      memref_desc.setStride(rewriter, loc, pos, stride_carried);
      // Update stride
      if (pos > 0) {
        stride_carried =
            rewriter.create<LLVM::MulOp>(loc, stride_carried, size);
      }
    }
    return memref_desc;
  }
  LogicalResult matchAndRewrite(
      memref::AllocOp alloc_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
  SymbolTable& symbol_table_;
};

// Emits LLVM IR to malloc a device memory.
LogicalResult ConvertMemRefAllocOpToDispatchOpPattern::matchAndRewrite(
    memref::AllocOp alloc_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  mlir::Operation* op = alloc_op.getOperation();
  Location loc = op->getLoc();

  // check address space
  auto memref = alloc_op.getResult();

  StringRef device = (placement_utils::isGpuMemRef(memref)) ? "gpu" : "cpu";
  MemRefType memref_type = memref.getType().cast<MemRefType>();

  // get ral context
  LLVMFuncOp parent_func = alloc_op->getParentOfType<LLVMFuncOp>();
  if (!parent_func) return failure();
  Value context_arg = parent_func.getArgument(0);

  // Get memref descriptor sizes.
  SmallVector<Value, 4> sizes;
  SmallVector<Value, 4> strides;
  Value sizeBytes;
  getMemRefDescriptorSizes(loc, memref_type,
                           llvm::to_vector<4>(adaptor.getOperands()), rewriter,
                           sizes, strides, sizeBytes);

  // create dispatch op
  auto dispatch_op = rewriter.create<disc_ral::DispatchOp>(
      loc, getVoidPtrType(), context_arg, sizeBytes, kMalloc, false, device);
  Value allocated_byte_ptr = dispatch_op.getResult(0);

  // Create the MemRef descriptor.
  MemRefDescriptor memRefDescriptor = CreateMemRefDescriptor(
      loc, rewriter, memref_type, allocated_byte_ptr, sizes);

  ModuleOp module = op->getParentOfType<ModuleOp>();
  rewriter.replaceOp(alloc_op, {memRefDescriptor});
  return success();
}

// A rewrite pattern to convert memref.dealloc operations into corresponding
// runtime wrapper calls (modeled by ral.dispatch ops)
// Converting:
//   memref.dealloc %0 : memref<?x?xf32, "gpu">
//     to
//   "disc_ral.dispatch"(%ctx, %1) {backend_config = "gpu", call_target_name
//   = "free", has_side_effect = false} : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
// then convert to llvm
class ConvertMemRefDeallocOpToDispatchOpPattern
    : public ConvertOpToLLVMPattern<memref::DeallocOp> {
 public:
  ConvertMemRefDeallocOpToDispatchOpPattern(LLVMTypeConverter& type_converter,
                                            SymbolTable& symbol_table)
      : ConvertOpToLLVMPattern<memref::DeallocOp>(type_converter),
        symbol_table_(symbol_table) {}

 private:
  LogicalResult matchAndRewrite(
      memref::DeallocOp dealloc_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
  SymbolTable& symbol_table_;
};

// Emits LLVM IR to dealloc a device memory.
LogicalResult ConvertMemRefDeallocOpToDispatchOpPattern::matchAndRewrite(
    memref::DeallocOp dealloc_op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  mlir::Operation* op = dealloc_op.getOperation();
  Location loc = op->getLoc();

  StringRef device =
      (placement_utils::isGpuMemRef(dealloc_op.memref())) ? "gpu" : "cpu";

  // get ral context
  LLVMFuncOp parent_func = dealloc_op->getParentOfType<LLVMFuncOp>();
  if (!parent_func) return failure();
  Value context_arg = parent_func.getArgument(0);

  // create dispatch op
  MemRefDescriptor memref(adaptor.memref());
  Value allocated_bytes_ptr = rewriter.create<LLVM::BitcastOp>(
      loc, getVoidPtrType(), memref.allocatedPtr(rewriter, loc));

  ModuleOp module = op->getParentOfType<ModuleOp>();
  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      op, llvm::None, context_arg, allocated_bytes_ptr, kFree, false, device);
  return success();
}

class ConvertCpuLaunchOpToDispatchOpPattern
    : public ConvertOpToLLVMPattern<disc_ral::CpuLaunchOp> {
 public:
  ConvertCpuLaunchOpToDispatchOpPattern(LLVMTypeConverter& type_converter,
                                        SymbolTable& symbol_table)
      : ConvertOpToLLVMPattern<disc_ral::CpuLaunchOp>(type_converter),
        symbol_table_(symbol_table) {}

 private:
  LogicalResult matchAndRewrite(
      disc_ral::CpuLaunchOp launchOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
  Value packArgs(ArrayRef<Value> arguments, ConversionPatternRewriter& rewriter,
                 Location& loc) const;
  LLVMFuncOp generatePackedKernel(disc_ral::CpuLaunchOp launchOp,
                                  ArrayRef<Value> arguments,
                                  ConversionPatternRewriter& rewriter) const;
  SymbolTable& symbol_table_;
};

// Packs the original inputs and outputs of the ral dispatch op to a uniform
// format.
//
// %struct = alloca(sizeof(struct { Parameters..., }))
// %array = alloca((NumParameters) * sizeof(void *))
// for (i : [0, NumParameters))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   llvm.store parameters[i], %fieldPtr
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// return %array
Value ConvertCpuLaunchOpToDispatchOpPattern::packArgs(
    ArrayRef<Value> arguments, ConversionPatternRewriter& rewriter,
    Location& loc) const {
  MLIRContext* ctx = rewriter.getContext();

  Type llvm_pointer_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  Type llvm_int32_type = IntegerType::get(ctx, 32);

  Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                 rewriter.getI32IntegerAttr(0));
  Value one = rewriter.create<LLVM::ConstantOp>(loc, llvm_int32_type,
                                                rewriter.getI32IntegerAttr(1));

  SmallVector<Type, 4> argument_types;
  for (auto argument : arguments) argument_types.push_back(argument.getType());

  auto struct_type =
      LLVM::LLVMStructType::getNewIdentified(ctx, StringRef(), argument_types);
  Value struct_ptr = rewriter.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(struct_type), one, /*alignment=*/0);
  Value array_size = rewriter.create<LLVM::ConstantOp>(
      loc, llvm_int32_type, rewriter.getI32IntegerAttr(argument_types.size()));
  Value array_ptr = rewriter.create<LLVM::AllocaOp>(
      loc, llvm_pointer_pointer_type, array_size, /*alignment=*/0);

  for (auto en : llvm::enumerate(argument_types)) {
    Value index = rewriter.create<LLVM::ConstantOp>(
        loc, llvm_int32_type, rewriter.getI32IntegerAttr(en.index()));
    Value field_ptr = rewriter.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(en.value()), struct_ptr,
        ArrayRef<Value>{zero, index});
    rewriter.create<LLVM::StoreOp>(loc, arguments[en.index()], field_ptr);

    Value element_ptr = rewriter.create<LLVM::GEPOp>(
        loc, llvm_pointer_pointer_type, array_ptr, index);
    Value casted =
        rewriter.create<LLVM::BitcastOp>(loc, llvm_pointer_type, field_ptr);
    rewriter.create<LLVM::StoreOp>(loc, casted, element_ptr);
  }

  return array_ptr;
}

LLVMFuncOp ConvertCpuLaunchOpToDispatchOpPattern::generatePackedKernel(
    disc_ral::CpuLaunchOp launchOp, ArrayRef<Value> arguments,
    ConversionPatternRewriter& rewriter) const {
  Location loc = launchOp->getLoc();
  MLIRContext* ctx = rewriter.getContext();

  StringRef kernelName = launchOp.callee();
  auto packedKernelName = (llvm::Twine("packed_") + kernelName).str();
  int numIvs =
      launchOp->getOperand(1).getType().cast<RankedTensorType>().getDimSize(0);

  Type llvm_pointer_type = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  Type llvm_pointer_pointer_type =
      LLVM::LLVMPointerType::get(llvm_pointer_type);
  Type llvm_int32_type = IntegerType::get(ctx, 32);
  Type llvm_int64_type = IntegerType::get(ctx, 64);
  Type llvm_int64_pointer_type = LLVM::LLVMPointerType::get(llvm_int64_type);

  // 1, collects arg types for packed kernel function
  SmallVector<Type> packedKernelArgTypes;
  // all ivs have index types. We map index type to i64 on CPU.
  for (int i = 0; i < 3; ++i)
    packedKernelArgTypes.push_back(llvm_int64_pointer_type);
  // packedArgs type
  packedKernelArgTypes.push_back(llvm_pointer_pointer_type);

  // 2, lock before creating the packed kernel function.
  OpBuilder::InsertionGuard guard(rewriter);
  OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
  ModuleOp module = launchOp->getParentOfType<ModuleOp>();
  rewriter.setInsertionPointToEnd(module.getBody());

  auto packedKernel = rewriter.create<LLVMFuncOp>(
      loc, packedKernelName,
      LLVM::LLVMFunctionType::get(getVoidType(), packedKernelArgTypes,
                                  /*isVarArg=*/false));
  // Update the symbol table
  symbol_table_.insert(packedKernel);
  packedKernel.addEntryBlock();

  Block* entry = &packedKernel.getBody().front();
  rewriter.setInsertionPoint(entry, entry->begin());

  SmallVector<Value> unpackedArgs;
  unpackedArgs.resize(5 * 3 + arguments.size());
  Value zero = rewriter.create<LLVM::ConstantOp>(loc, llvm_int64_type,
                                                 rewriter.getI64IntegerAttr(0));
  Value one = rewriter.create<LLVM::ConstantOp>(loc, llvm_int64_type,
                                                rewriter.getI64IntegerAttr(1));
  Value numIvsValue = rewriter.create<LLVM::ConstantOp>(
      loc, llvm_int64_type, rewriter.getI64IntegerAttr(1));
  // loop lower bound pointer
  unpackedArgs[1] = unpackedArgs[2] = packedKernel.getArgument(0);
  // offset, size, stride
  unpackedArgs[3] = zero;
  unpackedArgs[4] = numIvsValue;
  unpackedArgs[5] = one;
  // loop upper bound pointer
  unpackedArgs[6] = unpackedArgs[7] = packedKernel.getArgument(1);
  // offset, size, stride
  unpackedArgs[8] = zero;
  unpackedArgs[9] = numIvsValue;
  unpackedArgs[10] = one;
  // loop step pointer
  unpackedArgs[11] = unpackedArgs[12] = packedKernel.getArgument(2);
  // offset, size, stride
  unpackedArgs[13] = zero;
  unpackedArgs[14] = numIvsValue;
  unpackedArgs[15] = one;
  Value packedArgs = packedKernel.getArgument(3);
  for (auto& en : llvm::enumerate(arguments)) {
    Value index = rewriter.create<LLVM::ConstantOp>(
        loc, llvm_int32_type, rewriter.getI32IntegerAttr(en.index()));
    Value element_ptr = rewriter.create<LLVM::GEPOp>(
        loc, llvm_pointer_pointer_type, packedArgs, index);
    Value untyped_arg_ptr = rewriter.create<LLVM::LoadOp>(loc, element_ptr);
    auto arg_ptr_type = LLVM::LLVMPointerType::get(en.value().getType());
    Value typed_arg_ptr =
        rewriter.create<LLVM::BitcastOp>(loc, arg_ptr_type, untyped_arg_ptr);
    Value arg = rewriter.create<LLVM::LoadOp>(loc, typed_arg_ptr);
    int argIdx = (!en.index() ? 0 : en.index() + 15);
    unpackedArgs[argIdx] = arg;
  }
  rewriter.create<LLVM::CallOp>(
      loc, llvm::None, SymbolRefAttr::get(rewriter.getContext(), kernelName),
      unpackedArgs);
  rewriter.create<LLVM::ReturnOp>(loc, llvm::None);
  rewriter.restoreInsertionPoint(ip);
  return packedKernel;
}

// rewrite a cpuLaunch op to a ral dispatch op.
LogicalResult ConvertCpuLaunchOpToDispatchOpPattern::matchAndRewrite(
    disc_ral::CpuLaunchOp launchOp, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  mlir::Operation* op = launchOp.getOperation();
  Location loc = op->getLoc();
  MLIRContext* ctx = rewriter.getContext();

  // get ral context
  LLVMFuncOp parentFunc = op->getParentOfType<LLVMFuncOp>();
  if (!parentFunc) return failure();
  Value ralContext = parentFunc.getArgument(0);

  SmallVector<Value, 4> arguments = getTypeConverter()->promoteOperands(
      loc, launchOp.getOperands(), adaptor.getOperands(), rewriter);
  ArrayRef<Value> newOperands = arguments;

  // Basic idea is:
  // original cpu kernel:
  //   func @kernel(%ctx, %iv0, %iv1, ..., %arg0, %arg1, ...) {...}
  //
  // 0, pack kernel args (except iv args)
  //   %packedArgs = packArgs(%ctx, %arg0, %arg1, ...);
  //
  // 1, generate a packed kernel function for each cpu kernel.
  //    func @packed_kernel(%iv0, %iv1, ..., %packedArgs) {
  //      %ctx, %unpackedArgs... = unpackArgs(%packedArgs);
  //      call %kernel(%ctx, %iv0, %iv1, %unpackedArgs...)
  //    }
  //
  // 2, generate a disc_ral.dispatch op
  //  disc_ral.dispatch(%ral_ctx,
  //                    %lowerBound..., %upperBound..., %step...,
  //                    addressOf(%packed_kernel), %packed_args,
  //                    kRalCpuLaunch, ...)

  // 0, pack args for cpu kernel function:
  //  signature for cpuLaunchOp:
  //    cpu_launch(%ctx, %lower, %upper, %step, %unitWorkloadSizeHint,
  //    %otherArgsForKernel...);
  //  - the first arg of `CpuLaunchOp` is the ral ctx
  //  - the following three fields are lowerBound, upperBound, step of 1D memref
  //  Type.
  //    - each 1D memref type have 5 subfields (basePtr, dataPtr, offset, size,
  //    stride) after flatten.
  //  - unitWorkloadSizeHint has type index.
  int numPrefixArgs = 1 + 5 * 3 + 1;
  SmallVector<Value> argsToPack{newOperands.front()};
  for (Value arg : newOperands.drop_front(numPrefixArgs))
    argsToPack.push_back(arg);
  Value packedArgs = packArgs(argsToPack, rewriter, loc);

  // 1, generate a wrap function for each cpu kernel function
  auto packedKernel = generatePackedKernel(launchOp, argsToPack, rewriter);

  // 2, generate disc_ral.dispatch op
  Value funcPtr = rewriter.create<LLVM::AddressOfOp>(loc, packedKernel);
  Value untypedFuncPtr = rewriter.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8)), funcPtr);

  SmallVector<Value> ralDispatchOpArgs;
  // Create a name for each cpu kernel.
  StrT kernelNameVarName, kernelNameVarContent;
  kernelNameVarName.append("_cpu_kernel_");
  kernelNameVarName.append(launchOp.callee());
  kernelNameVarContent.append(launchOp.callee());
  kernelNameVarContent.push_back('\0');
  Value kernelNameValue = loadOrCreateGlobalString(
      rewriter, symbol_table_, launchOp, kernelNameVarName.str(),
      kernelNameVarContent.str());
  ralDispatchOpArgs.push_back(kernelNameValue);
  ralDispatchOpArgs.push_back(launchOp.lowerBound());
  ralDispatchOpArgs.push_back(launchOp.upperBound());
  ralDispatchOpArgs.push_back(launchOp.step());
  ralDispatchOpArgs.push_back(launchOp.unitWorkloadSizeHint());
  ralDispatchOpArgs.push_back(untypedFuncPtr);
  ralDispatchOpArgs.push_back(packedArgs);

  rewriter.replaceOpWithNewOp<disc_ral::DispatchOp>(
      launchOp, llvm::None, adaptor.ctx(), ralDispatchOpArgs, kRalCpuLaunch,
      false, "cpu");

  return success();
}

class DiscToLLVMPass : public DiscToLLVMPassBase<DiscToLLVMPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    SymbolTable symbol_table(m);

    // Populate type conversions.
    MLIRContext* ctx = m.getContext();
    LLVMTypeConverter type_converter(ctx);
    type_converter.addConversion([&](RalExecutionContextType type) {
      return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    });

    // Populate patterns.
    RewritePatternSet patterns(&getContext());
    mlir::arith::populateArithmeticToLLVMConversionPatterns(type_converter,
                                                            patterns);
    arith::populateArithmeticExpandOpsPatterns(patterns);
    // populateStdToLLVMConversionPatterns(type_converter, patterns);
    populateMemRefToLLVMConversionPatterns(type_converter, patterns);
    populateMathToLLVMConversionPatterns(type_converter, patterns);
    populateFuncToLLVMConversionPatterns(type_converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(type_converter, patterns);
    populateDiscToLLVMConversionPatterns(&type_converter, &symbol_table,
                                         &patterns);

    // Set target.
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<arith::ArithmeticDialect, gpu::GPUDialect,
                             disc_ral::RalDialect, math::MathDialect>();
    // Mark modules as legal.
    target.addLegalOp<ModuleOp, gpu::GPUModuleOp>();
    // Do not look into gpu modules, only consider host-side.
    target.markOpRecursivelyLegal<gpu::GPUModuleOp>();

    if (failed(applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // Finally, strip the GPU modules, as they are no longer needed.
    for (auto op : llvm::make_early_inc_range(m.getOps<gpu::GPUModuleOp>())) {
      op.erase();
    }
  }
};

static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter& rewriter,
                                           ModuleOp module) {
  auto* context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get(context, "printf");
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(Location loc, OpBuilder& builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module) {
  // Create the global at the entry of the module.
  LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value),
                                            /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(
      loc, IntegerType::get(builder.getContext(), 64),
      builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

// Converts a ral.dispatch_op to its llvm format.
struct PrintfToLLVMPattern : public ConvertOpToLLVMPattern<PrintfOp> {
  PrintfToLLVMPattern(LLVMTypeConverter& type_converter,
                      SymbolTable& symbol_table)
      : ConvertOpToLLVMPattern<PrintfOp>(type_converter),
        symbol_table_(symbol_table) {}

  LogicalResult matchAndRewrite(
      PrintfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;

 private:
  SymbolTable& symbol_table_;
};

// Sample:
//  SmallVector<Value, 4> buffer_args;
//  buffer_args.push_back(var_idx0);
//  buffer_args.push_back(var_idx1);
//  buffer_args.push_back(var_idx2);
//  auto lhloOp = b.create<lmhlo_disc::PrintfOp>(loc, llvm::None, buffer_args);
//  lhloOp->setAttr("format",
//                  b.getStringAttr("Debug idx0 %d idx1 %d "
//                                  "idx2 %d\n"));
LogicalResult PrintfToLLVMPattern::matchAndRewrite(
    PrintfOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const {
  Location loc = op->getLoc();
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();

  // Get a symbol reference to the printf function, inserting it if necessary.
  auto printfRef = getOrInsertPrintf(rewriter, parentModule);
  std::string key = llvm::Twine("DiscPrintf", op.format()).str();
  Value formatSpecCst =
      getOrCreateGlobalString(loc, rewriter, key, op.format(), parentModule);
  SmallVector<Value, 4> val_to_print{formatSpecCst};
  for (Value operand : adaptor.getOperands()) {
    val_to_print.push_back(operand);
  }
  rewriter.create<func::CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                val_to_print);
  rewriter.eraseOp(op);
  return success();
}

}  // namespace

void populateDiscToLLVMConversionPatterns(LLVMTypeConverter* converter,
                                          SymbolTable* symbol_table,
                                          RewritePatternSet* patterns) {
  // clang-format off
  patterns->insert<
      ConvertCpuLaunchOpToDispatchOpPattern,
      ConvertLaunchFuncOpToRalCallPattern,
      ConvertMemRefAllocOpToDispatchOpPattern,
      ConvertMemRefDeallocOpToDispatchOpPattern,
      DispatchOpToLLVMPattern,
      PrintfToLLVMPattern
    >(*converter, *symbol_table);
  // clang-format on
  patterns->insert<RemoveUselessUnrealizedConversionCastOp>(*converter);
}

std::unique_ptr<OperationPass<ModuleOp>> createDiscToLLVMPass() {
  return std::make_unique<DiscToLLVMPass>();
}

}  // namespace disc_ral
}  // namespace mlir
