// Copyright 2023 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "triton_disc/Conversion/TritonGPUToMLIRGPU.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton_disc/Conversion/Passes.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::triton;
using namespace mlir::triton_disc;
using namespace mlir::memref;

#define bitcast(val__, type__) b.create<LLVM::BitcastOp>(loc, type__, val__)

#define GEN_PASS_CLASSES
#include "triton_disc/Conversion/Passes.h.inc"

namespace {

struct GetProgramIdOpConversion
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      triton::GetProgramIdOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto i32_ty = IntegerType::get(rewriter.getContext(), 32);
    auto blockId = rewriter.create<mlir::gpu::BlockIdOp>(
        loc, rewriter.getIndexType(), mlir::gpu::Dimension::x);
    Value blockIdIdx =
        rewriter.create<::mlir::arith::IndexCastOp>(loc, i32_ty, blockId);
    rewriter.replaceAllUsesWith(op, blockIdIdx);
    blockIdIdx.getParentBlock()->dump();
    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

class TritonGPUToMLIRGPUTypeConverter : public LLVMTypeConverter {
 public:
  using TypeConverter::convertType;
  TritonGPUToMLIRGPUTypeConverter(MLIRContext* ctx, LowerToLLVMOptions& option,
                                  const DataLayoutAnalysis* analysis = nullptr)
      : LLVMTypeConverter(ctx, option, analysis) {
    addConversion([](Type type) { return type; });
    addConversion([&](IndexType type) -> llvm::Optional<Type> {
      return IntegerType::get(type.getContext(), 32);
    });
    addConversion([&](triton::PointerType type) -> std::optional<Type> {
      return convertTritonPointerType(type);
    });
  }
  Type convertTritonPointerType(triton::PointerType type);
};

Type TritonGPUToMLIRGPUTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  // Recursively translate pointee type
  return LLVM::LLVMPointerType::get(convertType(type.getPointeeType()),
                                    type.getAddressSpace());
}

MemRefDescriptor CreateMemRefDescriptor(Location loc, OpBuilder& rewriter,
                                        MemRefType memref_type,
                                        Value allocated_byte_ptr,
                                        ArrayRef<Value> sizes) {
  auto memref_desc = MemRefDescriptor::undef(rewriter, loc, memref_type);

  Value allocated_type_ptr = rewriter.create<LLVM::BitcastOp>(
      loc, FloatType::getF32(rewriter.getContext()), allocated_byte_ptr);
  memref_desc.setAllocatedPtr(rewriter, loc, allocated_type_ptr);
  memref_desc.setAlignedPtr(rewriter, loc, allocated_type_ptr);
  memref_desc.setConstantOffset(rewriter, loc, 0);
  if (memref_type.getRank() == 0) {
    return memref_desc;
  }

  // Compute strides and populate descriptor `size` and `stride` fields.
  Value stride_carried = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getIndexType(), rewriter.getI64IntegerAttr(1));
  for (int pos = sizes.size() - 1; pos >= 0; --pos) {
    Value size = sizes[pos];
    memref_desc.setSize(rewriter, loc, pos, size);
    memref_desc.setStride(rewriter, loc, pos, stride_carried);
    // Update stride
    if (pos > 0) {
      stride_carried = rewriter.create<LLVM::MulOp>(loc, stride_carried, size);
    }
  }
  return memref_desc;
}

scf::ParallelOp createParallelAndSetInsPt(OpBuilder& b, Location loc,
                                          SmallVectorImpl<Value>& vars,
                                          ArrayRef<Value> lbs,
                                          ArrayRef<Value> ubs,
                                          ArrayRef<Value> steps,
                                          ArrayRef<Value> initValues) {
  auto parOp = b.create<scf::ParallelOp>(loc, lbs, ubs, steps, initValues,
                                         /*bodyBuilderFn=*/nullptr);
  b.setInsertionPointToStart(parOp.getBody());
  vars.append(parOp.getInductionVars().begin(), parOp.getInductionVars().end());
  return parOp;
}

void emitInitLoops(OpBuilder& b, func::FuncOp root,
                   TypeConverter typeConverter) {
  Location loc = root.getLoc();
  // auto args = root.getArguments();
  // auto type = args[0].getType();
  // auto srcType = typeConverter.convertType(args[0].getType());
  // auto cased_arg0 = bitcast(args[0], srcType);

  // auto arg0 = CreateMemRefDescriptor(loc, b,
  // MemRefType::get({ShapedType::kDynamic},
  // FloatType::getF32(root.getContext())), args[0], {}); auto arg1 =
  // CreateMemRefDescriptor(loc, b, MemRefType::get({10000},
  // FloatType::getF32(root.getContext())), args[1], {});
  // create memref from llvm.ptr
  auto fakeCst = b.create<arith::ConstantIndexOp>(loc, 1000);
  SmallVector<Value, 4> dynShapes{fakeCst};
  auto arg0 = b.create<memref::AllocOp>(
                   loc,
                   MemRefType::get({ShapedType::kDynamic},
                                   FloatType::getF32(root.getContext())),
                   dynShapes)
                  .getResult();
  auto arg1 = b.create<memref::AllocOp>(
                   loc,
                   MemRefType::get({ShapedType::kDynamic},
                                   FloatType::getF32(root.getContext())),
                   dynShapes)
                  .getResult();
  auto result = b.create<memref::AllocOp>(
                     loc,
                     MemRefType::get({ShapedType::kDynamic},
                                     FloatType::getF32(root.getContext())),
                     dynShapes)
                    .getResult();

  // Value num_elements = b.create<memref::DimOp>(loc, first_arg, 0);
  // Value dim = arg0 = b.create<memref::DimOp>(loc, arg0, 0);
  Value num_elements = b.create<arith::ConstantIndexOp>(loc, 1000);
  Value block_size = b.create<arith::ConstantIndexOp>(loc, 256);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  // fake parallel to lower to gpu.function
  // for idx1 = 0 to num_elemnts step 1:
  //     %src1 = memref.load %arg0[%idx1] : memref<?xf32>
  //     %src2 = memref.load %arg1[%idx1] : memref<?xf32>
  //     %add = scf.addf %src1, %src2
  //     memref.store %add, %ret[%idx1] : memref<?xf32>
  SmallVector<Value, 2> vars1;
  auto forOp =
      createParallelAndSetInsPt(b, loc, /*vars=*/vars1, /*lbs=*/{zero},
                                /*ubs=*/{num_elements}, /*steps=*/{block_size},
                                /*initValues=*/{});

  SmallVector<Value, 2> vars2;
  auto for2 = createParallelAndSetInsPt(b, loc, /*vars=*/vars2, /*lbs=*/{zero},
                                        /*ubs=*/{block_size}, /*steps=*/{one},
                                        /*initValues=*/{});

  Value src1 = b.create<memref::LoadOp>(loc, arg0, zero);
  Value src2 = b.create<memref::LoadOp>(loc, arg1, zero);
  Value add = b.create<arith::AddFOp>(loc, src1, src2);
  b.create<memref::StoreOp>(loc, add, result, for2.getInductionVars()[0]);
}

void convertInputToLLVMPointer(func::FuncOp& func,
                               TritonGPUToMLIRGPUTypeConverter* typeConverter) {
  Location loc = func.getLoc();
  OpBuilder b(&func.getBody());
  Block* block = &func.getBody().front();
  FunctionType funcType = func.getFunctionType();
  size_t size = block->getArguments().size();
  SmallVector<Type, 4> types;
  for (size_t i = 0; i < size; ++i) {
    Value oldArgument = block->getArgument(i);
    Type newType = typeConverter->convertType(oldArgument.getType());
    types.push_back(newType);
    Value newArgument = block->addArgument(newType, loc);
    oldArgument.replaceAllUsesWith(newArgument);
  }
  for (size_t i = 0; i < size; ++i) block->eraseArgument(0);

  // update func signature
  func.setType(b.getFunctionType(types, {}));
}

class ConvertTritonGPUToMLIRGPU
    : public ConvertTritonGPUToMLIRGPUBase<ConvertTritonGPUToMLIRGPU> {
 public:
  explicit ConvertTritonGPUToMLIRGPU() {}

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ConversionTarget target(*context);
    ModuleOp mod = getOperation();
    target.addLegalDialect<mlir::gpu::GPUDialect>();
    mlir::LowerToLLVMOptions option(context);
    TritonGPUToMLIRGPUTypeConverter typeConverter(context, option, nullptr);

    SmallVector<func::FuncOp> funcs;
    mod.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
    auto func = funcs[0];
    OpBuilder b(&func.getBody());

    SmallVector<Operation*> ttOps;
    func.walk([&](Operation* op) { ttOps.push_back(op); });

    OpBuilder mb(mod);
    mb.setInsertionPointToStart(&mod.getBodyRegion().front());
    FunctionType funcType = b.getFunctionType({}, {});
    auto discFunc =
        mb.create<func::FuncOp>(mod.getLoc(), "disc_func", funcType);
    b.setInsertionPointToEnd(discFunc.addEntryBlock());
    OpBuilder discBuilder(&discFunc.getBody());
    discBuilder.create<func::ReturnOp>(discFunc.getLoc());

    // convertInputToLLVMPointer(discFunc, &typeConverter);
    discBuilder.setInsertionPointToStart(&discFunc.getBody().front());
    // note: emit a fake loopps to simulate tt ops lowering
    emitInitLoops(discBuilder, func, typeConverter);
    func.erase();

    RewritePatternSet patterns(context);
    // TODO(yancey): lowering load/store op in scf.parallel
    llvm::dbgs() << "dump func after init loops\n";
    mod.getOperation()->dump();
    patterns.insert<GetProgramIdOpConversion>(typeConverter, context);
    target.addIllegalOp<triton::GetProgramIdOp>();

    auto config = GreedyRewriteConfig();
    config.maxIterations = 100;
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config)))
      return signalPassFailure();
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::triton_disc::createTritonGPUToMLIRGPUPass() {
  return std::make_unique<::ConvertTritonGPUToMLIRGPU>();
}
