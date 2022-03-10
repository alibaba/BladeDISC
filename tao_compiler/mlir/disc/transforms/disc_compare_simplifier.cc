// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Analysis/shape_component_analysis.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/shape_utils.h"

namespace mlir {
namespace disc_ral {
namespace {

// This pattern analysis the operands of CmpIOp and try to infer the result
// at compile time with affine-expression analysis.
struct ArithComISimplifier : public OpRewritePattern<arith::CmpIOp> {
  explicit ArithComISimplifier(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto pred = op.getPredicate();

    ShapeComponentAnalysis shapeComponentAnalysis;
    auto lhsInfo = shapeComponentAnalysis.GetValueInfo(lhs);
    auto rhsInfo = shapeComponentAnalysis.GetValueInfo(rhs);
    if (!lhsInfo || ((*lhsInfo).size() != 1) || !rhsInfo ||
        ((*rhsInfo).size() != 1)) {
      return failure();
    }
    auto predict = compareSymbolicExpr((*lhsInfo)[0], (*rhsInfo)[0], pred);
    if (predict.hasValue()) {
      Value new_cmpi = rewriter.create<arith::ConstantIntOp>(
          loc, *predict, rewriter.getIntegerType(1));
      rewriter.replaceOp(op, new_cmpi);
      return success();
    }

    return failure();
  }

  struct SignState {
    enum State : int32_t {
      Unknown = 0,
      KnownNegative,
      KnownNotPositive,
      KnownZero,
      KnownNotNegative,
      KnownPositive
    };

    State state = Unknown;

    bool isKnownZero() { return state == KnownZero; }
    void setKnownZero() { state = KnownZero; }

    bool isKnownNegative() { return state == KnownNegative; }
    void setKnownNegative() { state = KnownNegative; }

    bool isKnownPositive() { return state == KnownPositive; }
    void setKnownPositive() { state = KnownPositive; }

    bool isKnownNotNegative() {
      return (state == KnownNotNegative) || isKnownPositive() || isKnownZero();
    }
    void setKnownNotNegative() {
      if (!isKnownZero() && !isKnownPositive()) {
        state = KnownNotNegative;
      }
    }

    bool isKnownNotPositive() {
      return (state == KnownNotPositive) || isKnownNegative() || isKnownZero();
    }
    void setKnownNotPositive() {
      if (!isKnownZero() && !isKnownNegative()) {
        state = KnownNotPositive;
      }
    }

    bool operator==(const SignState& other) { return other.state == state; }

    std::string str() {
      std::vector<std::string> state2Str = {
          "Unknown",   "KnownNegative",    "KnownNotPositive",
          "KnownZero", "KnownNotNegative", "KnownPositive"};
      return state2Str[state];
    }
  };

 public:
  using ShapeOrValueInfo = ShapeComponentAnalysis::ShapeOrValueInfo;
  using Symbol = ShapeComponentAnalysis::Symbol;
  using SymbolicExpr = ShapeComponentAnalysis::SymbolicExpr;
  using SymbolicExprsMap = ShapeComponentAnalysis::SymbolicExprsMap;

 private:
  SignState getSignState(const Symbol& symbol) const;
  SignState getSignState(const AffineExpr& expr,
                         const SmallVectorImpl<Symbol>& symbols) const;
  bool isSpecificConstant(const SymbolicExpr& symbolExpr, int64_t value) const;

  llvm::Optional<bool> compareSymbolicExpr(
      const SymbolicExpr& lhs, const SymbolicExpr& rhs,
      arith::CmpIPredicate predicate) const;
};

ArithComISimplifier::SignState ArithComISimplifier::getSignState(
    const ArithComISimplifier::Symbol& symbol) const {
  SignState state;
  int64_t index = symbol.index;
  // If the symbol is coming from a shape, it can't be negative. Also allow
  // results of shape_of, compute_reshape_shape, num_elements and broadcast.
  // This is correct, not complete.
  if (symbol.source.isShapeInfo()) {
    auto type = symbol.source.value().getType().dyn_cast<RankedTensorType>();
    if (type && type.getDimSize(index) > 0) {
      state.setKnownPositive();
    } else if (type && type.getDimSize(index) == 0) {
      state.setKnownZero();
    } else {
      state.setKnownNotNegative();
    }
  }
  Operation* op = symbol.source.value().getDefiningOp();
  if (op == nullptr) {
    return state;
  }
  if (llvm::isa<shape::ShapeOfOp, mhlo::ComputeReshapeShapeOp,
                shape::NumElementsOp, shape::BroadcastOp>(op)) {
    // TODO: double-check shape.broadcast. If the semantics allow negative
    // value, check whether the broadcast is a producer of mhlo.bcast/reshape
    // op.
    state.setKnownNotNegative();
  } else if (auto select = dyn_cast_or_null<arith::SelectOp>(op)) {
    auto trueValue = select.getTrueValue();
    auto falseValue = select.getFalseValue();
    ShapeComponentAnalysis shapeComponentAnalysis;
    auto trueInfo = shapeComponentAnalysis.GetValueInfo(trueValue);
    auto falseInfo = shapeComponentAnalysis.GetValueInfo(falseValue);
    if (trueInfo && ((*trueInfo).size() == 1) && falseInfo &&
        ((*falseInfo).size() == 1)) {
      auto trueSymbolicExpr = (*trueInfo)[0];
      auto falseSymbolicExpr = (*falseInfo)[0];
      auto trueState =
          getSignState(trueSymbolicExpr.expr, trueSymbolicExpr.symbols);
      auto falseState =
          getSignState(falseSymbolicExpr.expr, falseSymbolicExpr.symbols);
      if (trueState == falseState) {
        state = trueState;
      } else if (trueState.isKnownNotNegative() &&
                 falseState.isKnownNotNegative()) {
        // The state may be different, but are all not negative.
        state.setKnownNotNegative();
      } else if (trueState.isKnownNotPositive() &&
                 falseState.isKnownNotPositive()) {
        // The state may be different, but are all not positive.
        state.setKnownNotPositive();
      }
    }
  }

  return state;
}

ArithComISimplifier::SignState ArithComISimplifier::getSignState(
    const AffineExpr& expr,
    const SmallVectorImpl<ArithComISimplifier::Symbol>& symbols) const {
  // TODO: make use of simplifyAffineExpr. Carefully deal with expr.getPosition
  // in symbols.
  SignState state;
  if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
    state = getSignState(symbols[symExpr.getPosition()]);
  } else if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    if (constExpr.getValue() > 0) {
      state.setKnownPositive();
    } else if (constExpr.getValue() < 0) {
      state.setKnownNegative();
    } else {
      state.setKnownZero();
    }
  } else if (auto bexpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    auto lhsState = getSignState(bexpr.getLHS(), symbols);
    auto rhsState = getSignState(bexpr.getRHS(), symbols);
    switch (bexpr.getKind()) {
      case AffineExprKind::Mul:
      case AffineExprKind::FloorDiv:
      case AffineExprKind::CeilDiv: {
        bool knownSameSign = false;
        knownSameSign |=
            (lhsState.isKnownPositive() & rhsState.isKnownPositive());
        knownSameSign |=
            (lhsState.isKnownNegative() & rhsState.isKnownNegative());
        if (knownSameSign) {
          state.setKnownPositive();
        }

        bool knownDiffSign = false;
        knownDiffSign |=
            (lhsState.isKnownPositive() & rhsState.isKnownNegative());
        knownDiffSign |=
            (lhsState.isKnownNegative() & rhsState.isKnownPositive());
        if (knownDiffSign) {
          state.setKnownNegative();
        }

        bool knownSameNotSign = false;
        knownSameNotSign |=
            (lhsState.isKnownNotPositive() & rhsState.isKnownNotPositive());
        knownSameNotSign |=
            (lhsState.isKnownNotNegative() & rhsState.isKnownNotNegative());
        if (knownSameNotSign) {
          state.setKnownNotNegative();
        }

        if (lhsState.isKnownZero() || (bexpr.getKind() == AffineExprKind::Mul &&
                                       rhsState.isKnownZero())) {
          state.setKnownZero();
        }
      } break;
      case AffineExprKind::Add: {
        SignState effectiveState = lhsState;
        if (lhsState.isKnownZero()) {
          state = rhsState;
        } else if (rhsState.isKnownZero()) {
          state = lhsState;
        }

        if ((lhsState.isKnownPositive() && rhsState.isKnownNotNegative()) ||
            (lhsState.isKnownNotNegative() && rhsState.isKnownPositive())) {
          state.setKnownPositive();
        }
        if ((lhsState.isKnownNegative() && rhsState.isKnownNotPositive()) ||
            (lhsState.isKnownNotPositive() && rhsState.isKnownNegative())) {
          state.setKnownNegative();
        }

        if (lhsState.isKnownNotNegative() && rhsState.isKnownNotNegative()) {
          state.setKnownNotNegative();
        }
        if (lhsState.isKnownNotPositive() && rhsState.isKnownNotPositive()) {
          state.setKnownNotPositive();
        }
      } break;
      case AffineExprKind::Mod: {
        // Note that rhs of Mod is always positive.
        if (lhsState.isKnownZero()) {
          state.setKnownZero();
        }
        if (lhsState.isKnownNotNegative()) {
          state.setKnownNotNegative();
        }
        if (lhsState.isKnownNotPositive()) {
          state.setKnownNotPositive();
        }
      } break;
    }
  }

  return state;
}

bool ArithComISimplifier::isSpecificConstant(
    const ArithComISimplifier::SymbolicExpr& symbolExpr, int64_t value) const {
  return symbolExpr.isConstant(value);
}

llvm::Optional<bool> ArithComISimplifier::compareSymbolicExpr(
    const ArithComISimplifier::SymbolicExpr& lhs,
    const ArithComISimplifier::SymbolicExpr& rhs,
    arith::CmpIPredicate predicate) const {
  auto lhsState = getSignState(lhs.expr, lhs.symbols);
  auto rhsState = getSignState(rhs.expr, rhs.symbols);
  llvm::Optional<bool> predResult = llvm::None;

  bool knownNotEq =
      (lhsState.isKnownNegative() && rhsState.isKnownNotNegative()) ||
      (lhsState.isKnownPositive() && rhsState.isKnownNotPositive()) ||
      (lhsState.isKnownNotNegative() && rhsState.isKnownNegative()) ||
      (lhsState.isKnownNotPositive() && rhsState.isKnownPositive());
  bool knownEq = (lhs == rhs);
  bool knownLt =
      (lhsState.isKnownNegative() && rhsState.isKnownNotNegative()) ||
      (lhsState.isKnownNotPositive() && rhsState.isKnownPositive());
  bool knownGt =
      (lhsState.isKnownNotNegative() && rhsState.isKnownNegative()) ||
      (lhsState.isKnownPositive() && rhsState.isKnownNotPositive());
  switch (predicate) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne: {
      if (knownNotEq) {
        predResult = (predicate == arith::CmpIPredicate::eq) ? false : true;
      } else if (knownEq) {
        predResult = (predicate == arith::CmpIPredicate::eq) ? true : false;
      }
    } break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule: {
      if (knownLt) {
        predResult = true;
      } else if (knownGt) {
        predResult = false;
      }
      if (knownEq && (predicate == arith::CmpIPredicate::sle) ||
          (predicate == arith::CmpIPredicate::ule)) {
        predResult = true;
      }
    } break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge: {
      if (knownLt) {
        predResult = false;
      } else if (knownGt) {
        predResult = true;
      }
      if (knownEq && (predicate == arith::CmpIPredicate::sle) ||
          (predicate == arith::CmpIPredicate::ule)) {
        predResult = true;
      }
    } break;
  }

  return predResult;
}

// TODO: CmpFOP simplifier

struct DiscCompareSimplifierPass
    : public DiscCompareSimplifierPassBase<DiscCompareSimplifierPass> {
  void runOnOperation() override {
    FuncOp func = getOperation();
    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ArithComISimplifier>(ctx);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDiscCompareSimplifierPass() {
  return std::make_unique<DiscCompareSimplifierPass>();
}

}  // namespace disc_ral
}  // namespace mlir