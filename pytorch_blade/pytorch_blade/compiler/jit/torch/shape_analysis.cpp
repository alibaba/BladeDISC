/* From PyTorch:
 *
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon
 * Bottou, Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research
 * Institute (Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute
 * (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 */

#define CAFFE2_LOG_THRESHOLD 0
#include "pytorch_blade/compiler/jit/torch/shape_analysis.h"
#include "pytorch_blade/common_utils/macros.h"

#include <c10/util/Exception.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/exception_message.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_strided.h>
#endif

#include <exception>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "c10/util/logging_is_not_google_glog.h"
#include "op_registry.h"

namespace c10 {
inline std::vector<size_t> irange(size_t start, size_t end) {
  std::vector<size_t> range(end - start);
  std::iota(range.begin(), range.end(), start);
  return range;
}

inline std::vector<size_t> irange(size_t end) {
  std::vector<size_t> range(end);
  std::iota(range.begin(), range.end(), 0);
  return range;
}
} // namespace c10

namespace torch {
namespace blade {
using namespace torch::jit;
using namespace c10;

bool PropagateTensorShapeOnNode(Node* node, bool insert_expands);

ShapeSymbol getSymDimSize(TensorTypePtr type, int64_t dim) {
  if (auto rank = type->symbolic_sizes().rank()) {
    dim = at::maybe_wrap_dim(dim, *rank, false);
    if (auto sizes = type->symbolic_sizes().sizes()) {
      auto dimSize = (*sizes)[dim];
      if (dimSize.is_static())
        return ShapeSymbol::fromStaticSize(dimSize.static_size());
    }
  }
  return ShapeSymbol::newSymbol();
}

c10::Device getDeviceFromValue(Value* value) {
#ifdef TORCH_BLADE_BUILD_WITH_CUDA
  c10::Device device = at::kCUDA;
#else
  c10::Device device = at::kCPU;
#endif
  if (value == nullptr)
    return device;
  if (value->node()->kind() == prim::device) {
    auto inpNode = value->node();
    auto inpTy = inpNode->input()->type()->cast<TensorType>();
    if (inpTy && inpTy->device()) {
      device = *inpTy->device();
    }
    return device;
  }

  auto maybe_device_option = ::torch::jit::toIValue(value);
  if (maybe_device_option && maybe_device_option->isDevice()) {
    device = maybe_device_option->toDevice();
  }
  return device;
}

c10::optional<at::ScalarType> getScalarTypeFromValue(Value* value) {
  c10::optional<at::ScalarType> scalarType;
  if (value == nullptr)
    return scalarType;
  if (value->node()->kind() == prim::dtype) {
    auto inpNode = value->node();
    auto inpTy = inpNode->input()->type()->cast<TensorType>();

    if (inpTy && inpTy->scalarType()) {
      scalarType = *inpTy->scalarType();
    }
    return scalarType;
  }

  auto maybe_scalar_type_option = ::torch::jit::toIValue(value);
  if (maybe_scalar_type_option && !maybe_scalar_type_option->isNone()) {
    scalarType = maybe_scalar_type_option->toScalarType();
  }
  return scalarType;
}

bool mergeTypes(
    at::ArrayRef<Value*> lhs,
    at::ArrayRef<Value*> rhs,
    at::ArrayRef<Value*> outputs) {
  AT_ASSERT(lhs.size() == rhs.size() && rhs.size() == outputs.size());
  bool changed = false;
  for (const auto i : c10::irange(lhs.size())) {
    auto old_output_type = outputs[i]->type();
    auto new_type =
        unifyTypes(lhs[i]->type(), rhs[i]->type(), /*default_to_union=*/true);
    AT_ASSERT(new_type);
    outputs[i]->setType(*new_type);
    if (*old_output_type != *outputs[i]->type())
      changed = true;
  }
  return changed;
}

void applyTypes(at::ArrayRef<Value*> src, at::ArrayRef<Value*> dst) {
  AT_ASSERT(src.size() == dst.size());
  for (const auto i : c10::irange(src.size())) {
    dst[i]->setType(src[i]->type());
  }
}

void PropertyPropBase::propagateBlock(Block* block, bool insert_expands) {
  for (Node* node : block->nodes()) {
    try {
      propagateNode(node, insert_expands);
    } catch (std::exception& e) {
      ErrorReport errMsg(node->sourceRange());
      errMsg << ExceptionMessage(e)
             << "failed shape propagation in this context. "
             << "The above operation: \n"
             << *node << "The inputs are:\n";
      for (auto inp : node->inputs()) {
        errMsg << *inp->type() << " from " << *(inp->node());
      }
      throw errMsg;
    }
  }
}

void PropertyPropBase::processIf(Node* node) {
  auto then_block = node->blocks().at(0);
  auto else_block = node->blocks().at(1);
  propagateBlock(then_block);
  propagateBlock(else_block);
  mergeTypes(then_block->outputs(), else_block->outputs(), node->outputs());
}

void PropertyPropBase::processLoop(Node* node) {
  LoopView loop(node);
  // propagate counter type
  loop.currentTripCount()->setType(loop.maxTripCount()->type());
  applyTypes(loop.carriedInputs(), loop.bodyCarriedInputs());

  do {
    propagateBlock(loop.bodyBlock(), /*insert_expands=*/false);
    // note: inserting expands is unsafe at this point, we don't know
    // if the types are stable yet, so the arguments to expand may change
  } while (mergeTypes(
      loop.bodyCarriedInputs(),
      loop.bodyCarriedOutputs(),
      loop.bodyCarriedInputs()));

  // now that the types are stable, we can insert the expands
  propagateBlock(loop.bodyBlock(), /*insert_expands=*/true);
  applyTypes(loop.bodyCarriedInputs(), loop.carriedOutputs());
}

void PropertyPropBase::setUnshapedType(Value* o) {
  auto type = o->type();
  TypePtr withDimOrUnshapedType;
  if (TensorTypePtr tt = type->cast<TensorType>()) {
    withDimOrUnshapedType = tt->withDim(tt->sizes().size());
  } else {
    withDimOrUnshapedType = unshapedType(type);
  }
  o->setType(withDimOrUnshapedType);
}

void PropertyPropBase::setUnshapedType(Node* node) {
  for (auto o : node->outputs()) {
    setUnshapedType(o);
  }
}

namespace prim {
using namespace ::c10::prim;
}

#define SHAPE_ASSERT(cond) \
  if (!(cond))             \
  throw propagation_error()

namespace {

bool isValidArgumentForRunning(Value* v) {
  // allow constants
  if (toIValue(v))
    return true;
  if (TensorTypePtr tt = v->type()->cast<TensorType>()) {
    if (!tt->scalarType()) {
      return false;
    }
    return !at::isIntegralType(*tt->scalarType(), /*includeBool=*/false);
  }
  return v->type()->isSubtypeOf(FloatType::get()) ||
      v->type()->isSubtypeOf(NumberType::get());
}

bool isValidReturnForRunning(Value* v) {
  return v->type()->isSubtypeOf(TensorType::get()) ||
      v->type()->isSubtypeOf(NumberType::get());
}

bool containsTensorType(const TypePtr& t) {
  auto n_contained = t->containedTypes().size();
  if (n_contained == 1) {
    return t->containedTypes().at(0)->isSubtypeOf(TensorType::get());
  } else if (n_contained > 1) {
    return std::any_of(
        t->containedTypes().begin(),
        t->containedTypes().end(),
        containsTensorType);
  }
  return false;
}

// for each node in the schema with type Tensor, extract the T type
// returns c10::nullopt if any Tensor in the schema does not have a known
// shape ignores non-tensor in the list of inputs
c10::optional<std::vector<TensorTypePtr>> gatherTensorTypes(
    Node* node,
    bool complete = false) {
  std::vector<TensorTypePtr> tensor_types;
  auto schema_opt = node->maybeSchema();
  if (!schema_opt) {
    return c10::nullopt;
  }
  auto& schema = *schema_opt;
  auto& args = schema.arguments();
  // can't handle varargs primitives because we don't know what should be a
  // Tensor
  if (schema.is_vararg()) {
    return c10::nullopt;
  }
  for (const auto i : c10::irange(args.size())) {
    if (args[i].type()->isSubtypeOf(ListType::ofTensors())) {
      return c10::nullopt;
    } else if (args[i].type()->isSubtypeOf(TensorType::get())) {
      if (auto type = node->input(i)->type()->cast<TensorType>()) {
        if (complete && !type->isComplete()) {
          return c10::nullopt;
        }
        tensor_types.push_back(type);
      } else {
        return c10::nullopt;
      }
    } else /* non-tensor type */ {
      continue;
    }
  }
  return tensor_types;
}

int64_t wrapDim(int64_t dim, at::IntArrayRef sizes) {
  if (dim < 0) {
    dim += (int64_t)sizes.size();
  }
  return dim;
}

c10::ScalarType unionScalarTypes(
    c10::ScalarType original,
    c10::ScalarType next) {
  if (original == c10::ScalarType::Undefined) {
    return next;
  } else {
    return c10::promoteTypes(original, next);
  }
}

TypePtr getDType(const Type& type) {
  auto tensorTy = type.expect<TensorType>();
  if (!tensorTy)
    return nullptr;
  auto inputDtype = tensorTy->scalarType();
  if (!inputDtype)
    return nullptr;

  if (isFloatingType(*inputDtype)) {
    return FloatType::get();
  } else if (isIntegralType(*inputDtype, /*includeBool*/ false)) {
    return IntType::get();
  } else if (isIntegralType(*inputDtype, /*includeBool*/ true)) {
    return BoolType::get();
  }
  return nullptr;
}

// Promotes result types for arithmetic operations on Tensor operands using
// new type promotion logic. See tensor_attributes.rst for details.
// This doesn't handle the case of arithmetic ops with Scalar arguments (when
// `Tensor.getUnsafeTensorImpl()->is_wrapped_nubmer()` would return true)
c10::optional<c10::ScalarType> getPromotedTypeForArithmeticOp(Node* node) {
  c10::ScalarType dimmed = c10::ScalarType::Undefined;
  c10::ScalarType zerodim = c10::ScalarType::Undefined;
  // binary arithmetic ops, more than 2 args is alpha.
  for (const auto i : c10::irange(2)) {
    auto dtt = node->inputs()[i]->type()->expect<TensorType>();
    auto inputDtype = dtt->scalarType();
    if (!dtt || !inputDtype) {
      return c10::nullopt;
    }
    if (dtt->dim() && *dtt->dim() > 0) {
      dimmed = unionScalarTypes(dimmed, *inputDtype);
    } else if (!isFloatingType(dimmed)) {
      // if no dimensions
      zerodim = unionScalarTypes(zerodim, *inputDtype);
    }
  }
  // if a tensor with dimensions is already of the highest category, don't
  // need to check zero-dim tensors.
  if (isFloatingType(dimmed)) {
    return dimmed;
  }
  // int_tensor * zero_dim_floating -> floating_tensor
  if (isIntegralType(dimmed, false) && isFloatingType(zerodim)) {
    return zerodim;
  }
  // bool_tensor * non_bool_scalar -> non_bool_tensor
  if (c10::ScalarType::Bool == dimmed &&
      c10::ScalarType::Undefined != zerodim) {
    return zerodim;
  }
  // types of dimensioned tensors generally take precedence over zero-dim
  // tensors if not promoting due to category. e.g.:
  // int_tensor * long -> int_tensor
  if (c10::ScalarType::Undefined != dimmed) {
    return dimmed;
  }

  // no dimmed tensors. e.g. zero_dim_tensor + zero_dim_tensor.
  return zerodim;
}

class ShapePropagator : public PropertyPropBase {
 public:
  explicit ShapePropagator(const std::shared_ptr<Graph>& graph)
      : PropertyPropBase(graph) {}

 private:
  IValue representativeValue(Value* v) {
    TypePtr type_ = v->type();
    // if the value is actually constant, just use it!
    if (auto iv = toIValue(v)) {
      return *iv;
    }
    if (TensorTypePtr type = type_->cast<TensorType>()) {
      if (type->isComplete()) {
        at::DeviceGuard device_guard(*type->device());
        return at::empty_strided(
                   *type->sizes().concrete_sizes(),
                   *type->strides().concrete_sizes(),
                   at::TensorOptions(*type->device())
                       .dtype(*type->scalarType()))
            .zero_();
      }
      // fallthrough
    } else if (type_->isSubtypeOf(FloatType::get())) {
      return 0.f;
    } else if (type_->isSubtypeOf(IntType::get())) {
      return 0;
    }
    // we should not get here because isValidArgumentForRunning should have
    // prevented it
    std::stringstream ss;
    ss << "unable to create representative value for: " << type_->str()
       << ". File a bug report";
    throw std::runtime_error(ss.str());
  }

  void broadcastBinary(
      Node* node,
      std::vector<TensorTypePtr>& types,
      size_t idx1,
      size_t idx2) {
    auto expected_size = at::infer_size(
        *types[idx1]->sizes().concrete_sizes(),
        *types[idx2]->sizes().concrete_sizes());
    auto broadcast = [&](size_t input_idx) {
      TensorTypePtr input_type = types.at(input_idx);
      if (input_type->sizes() == expected_size)
        return;
      auto graph = node->owningGraph();
      WithInsertPoint point_guard{node};
      Node* expand = graph
                         ->create(
                             aten::expand,
                             {node->inputs().at(input_idx),
                              graph->insertConstant(expected_size),
                              graph->insertConstant(false)})
                         ->insertBefore(node);
      propagateNode(expand);
      node->replaceInput(input_idx, expand->output());
    };
    broadcast(idx1);
    broadcast(idx2);
    types[0] = node->inputs().at(idx1)->type()->expect<TensorType>();
    types[1] = node->inputs().at(idx2)->type()->expect<TensorType>();
  }

  SchemaSet cannot_propagate_shape_by_running_it = {
      "aten::inverse(Tensor self) -> Tensor",
  };

  bool canPropagateShapeByRunningIt(Node* node) {
    if (cannot_propagate_shape_by_running_it.hasMember(*node)) {
      return false;
    }

    bool valid_args = std::all_of(
        node->inputs().begin(),
        node->inputs().end(),
        isValidArgumentForRunning);
    if (!valid_args)
      return false;

    bool valid_returns = std::all_of(
        node->outputs().begin(),
        node->outputs().end(),
        isValidReturnForRunning);
    if (!valid_returns)
      return false;

    return true;
  }

  // If there's no Tensor in outputs, e.g float / float,
  // we don't need to propagate shape.
  bool DoesntRefineOutputs(Node* node) {
    auto outputs = node->outputs();
    for (auto& out : outputs) {
      if (containsTensorType(out->type())) {
        return false;
      }
    }
    return true;
  }

  bool PropagateShapeOnNodeByRunningIt(Node* node, Operation op = nullptr) {
    if (!canPropagateShapeByRunningIt(node))
      return false;

    if (!op)
      op = node->getOperation();

    Stack stack;

    for (auto input : node->inputs()) {
      stack.push_back(representativeValue(input));
    }

    // XXX: we're not catching any exceptions from the op for now. This
    // is to uncover any mistakes we could make when editing this code,
    // and eventually it shouldn't matter, because this phase should be
    // preceded by schema checking.
#if PYTORCH_VERSION_GE(1, 10)
    op(stack);
#else
    op(&stack);
#endif
    AT_ASSERT(stack.size() == node->outputs().size());
    for (const auto i : c10::irange(stack.size())) {
      // some ops may have mixed tensor/primitive outputs
      // for primitives, we don't need to change the type because it is already
      // its most constrained form.
      auto tensor_type = node->outputs()[i]->type()->cast<TensorType>();
      if (stack[i].isTensor() && tensor_type) {
        // gradient information isn't always available or part of represenative
        // inputs, maintain original grad property
        auto tensor_grad = tensor_type->requiresGrad();
        node->outputs()[i]->setType(TensorType::create(stack[i].toTensor())
                                        ->withRequiresGrad(tensor_grad));
      }
    }
    return true;
  }

  void PropagateCatShape(Node* cat_node) {
    static const auto propagate_complete =
        [](Node* node, at::ArrayRef<Value*> tensors) -> bool {
      auto input_types =
          fmap(tensors, [](Value* v) { return v->type()->cast<TensorType>(); });
      if (!std::all_of(
              input_types.begin(),
              input_types.end(),
              [](const TensorTypePtr& tp) {
                return tp != nullptr && tp->isComplete();
              })) {
        return false;
      }
      if (!node->is_constant(attr::dim))
        return false;
      std::vector<int64_t> sizes = *input_types[0]->sizes().concrete_sizes();
      const int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      const int64_t ndim = (int64_t)sizes.size();

      if (dim < 0 || dim >= ndim)
        return false;

      sizes[dim] = 0;
      for (auto& tp : input_types) {
        auto tp_sizes = tp->sizes().concrete_sizes().value();
        if (sizes.size() != tp_sizes.size())
          return false;
        for (const auto i : c10::irange(ndim)) {
          if (sizes[i] != tp_sizes[i] && i != dim) {
            return false;
          }
        }
        sizes[dim] += tp_sizes[dim];
      }
      node->output()->setType(input_types[0]->withSizes(sizes));
      return true;
    };
    static const auto propagate = [](Node* node,
                                     at::ArrayRef<Value*> tensors) -> bool {
      for (Value* v : tensors) {
        if (auto type = v->type()->cast<TensorType>()) {
          node->output()->setType(type->dimensionedOnly());
          return true;
        }
      }
      return false;
    };
    auto list_node =
        ((cat_node->kind() == prim::FusedConcat)
             ? cat_node
             : cat_node->namedInput(attr::tensors)->node());
    if (list_node->kind() == prim::ListConstruct ||
        cat_node->kind() == prim::FusedConcat) {
      auto tensors = list_node->inputs();
      if (!tensors.empty()) {
        // NOLINTNEXTLINE(bugprone-branch-clone)
        if (propagate_complete(cat_node, tensors)) {
          return;
        } else if (propagate(cat_node, tensors)) {
          return;
        }
      }
    }
    setUnshapedType(cat_node);
  }

#if PYTORCH_VERSION_LE(1, 10)
  c10::optional<c10::ScalarType> tryScalarTypeFromJitType(
      const c10::Type& type) {
    if (type == *FloatType::get()) {
      return at::typeMetaToScalarType(c10::get_default_dtype());
    } else if (type == *IntType::get()) {
      return at::ScalarType::Long;
    } else if (type == *BoolType::get()) {
      return at::ScalarType::Bool;
    }
    return c10::nullopt;
  }
#endif

  void propagateTorchTensorShape(Node* node) {
    auto input_type = node->inputs().at(0)->type();

    size_t dims = 0;
    auto input_base_type = input_type;
    auto list_type = input_type->cast<ListType>();
    while (list_type) {
      dims++;
      input_base_type = list_type->getElementType();
      list_type = input_base_type->cast<ListType>();
    }

    at::optional<at::ScalarType> default_type =
        tryScalarTypeFromJitType(*input_base_type);
    if (auto grad_index = node->schema().argumentIndexWithName("dtype")) {
      auto inp = toIValue(node->inputs().at(*grad_index));
      if (inp == c10::nullopt) {
        return;
      } else if (!inp->isNone()) {
        default_type = inp->toScalarType();
      }
    }

    auto device = getDeviceFromValue(node->namedInput(attr::device));
    node->output()->setType(TensorType::create(
        default_type, device, dims, /*requires_grad=*/c10::nullopt));
  }

  void propagateNode(Node* node, bool insert_expands = true) override {
    // These don't require the types, and have complicated schema. Return early
    // after we process them.
    switch (node->kind()) {
      case prim::If:
        return processIf(node);
      case prim::Loop: {
        return processLoop(node);
      }
      case aten::Bool:
      case aten::Int:
      case aten::Float:
      case aten::FloatImplicit:
      case aten::IntImplicit:
      case aten::size:
      case prim::device:
      case prim::dtype:
      case prim::RaiseException:
      case aten::dim:
      case aten::append:
        return; // correct type is already set
      case aten::item:
      case aten::ScalarImplicit: {
        if (auto dtype = getDType(*node->input()->type())) {
          node->output()->setType(dtype);
        }
        return;
      }
      case prim::NumToTensor: {
        TypePtr typ = node->input()->type();
        if (typ->isSubtypeOf(IntType::get()) ||
            typ->isSubtypeOf(BoolType::get())) {
          node->output()->setType(TensorType::create(
              at::kLong, at::kCPU, 0, /*requires_grad=*/c10::nullopt));
        } else if (node->input()->type()->isSubtypeOf(FloatType::get())) {
          node->output()->setType(TensorType::create(
              at::kDouble, at::kCPU, 0, /*requires_grad=*/c10::nullopt));
        }
        return;
      }
      case aten::tensor:
        if (node->matches(
                "aten::tensor(t[] data, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)")) {
          propagateTorchTensorShape(node);
          PropagateTensorShapeOnNode(node, insert_expands);
          return;
        }
      case aten::as_tensor: {
        // as_tensor has an overloaded schema and can either have a tensor or
        // a list as the first input, if the input is a tensor, we delegate
        // the shape propagation in PropagateTensorShapeOnNode
        if (node->inputs().at(0)->type()->isSubtypeOf(TensorType::get())) {
          break;
        }
        return propagateTorchTensorShape(node);
      }
      case prim::TupleConstruct: {
        // We refresh the tuple type, because the input types could have been
        // refined.
        auto orig_type = node->output()->type()->expect<TupleType>();
        auto new_types =
            fmap(node->inputs(), [](Value* v) { return v->type(); });
        node->output()->setType(
            orig_type->createWithContained(std::move(new_types)));
        return;
      }
      case prim::TupleUnpack: {
        auto block_inputs = node->owningBlock()->inputs();
        auto it =
            std::find(block_inputs.begin(), block_inputs.end(), node->input());
        if (it != block_inputs.end()) {
          setUnshapedType(node);
          return;
        }

        auto tuple_type = node->input()->type()->cast<TupleType>();
        AT_ASSERT(
            tuple_type &&
            tuple_type->elements().size() == node->outputs().size());
        auto elems = tuple_type->elements();
        for (size_t i = 0; i < node->outputs().size(); ++i) {
          node->output(i)->setType(elems[i]);
        }
        return;
      }
      case prim::ListUnpack: {
        auto input_node = node->input()->node();
        if (input_node->matches(
                "aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]") ||
            input_node->matches(
                "aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]") ||
            input_node->matches(
                "aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]")) {
          if (auto self_type =
                  input_node->input(0)->type()->cast<TensorType>()) {
            auto sizes_opt = self_type->symbolic_sizes().sizes();
            auto dim_opt = input_node->get<int64_t>(attr::dim);
            if (!(sizes_opt && dim_opt))
              return;

            std::vector<c10::ShapeSymbol> new_sizes = sizes_opt.value();
            int64_t input_rank = new_sizes.size();
            int64_t dim =
                at::maybe_wrap_dim(dim_opt.value(), input_rank, false);
            if (input_node->matches(
                    "aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]")) {
              new_sizes.erase(new_sizes.begin() + dim);
            } else {
              new_sizes[dim] = ShapeSymbol::newSymbol();
            }

            for (size_t i = 0; i < node->outputs().size(); ++i) {
              if (auto type = node->output(i)->type()->cast<TensorType>()) {
                c10::ShapeSymbol dim_value = ShapeSymbol::newSymbol();
                if (input_node->matches(
                        "aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]") &&
                    sizes_opt.value()[dim].is_static()) {
                  int64_t split_size =
                      input_node->get<int64_t>(attr::split_size).value();
                  if (split_size * (i + 1) > self_type->sizes()[dim].value()) {
                    split_size =
                        self_type->sizes()[dim].value() - split_size * i;
                  }
                  new_sizes[dim] = ShapeSymbol::fromStaticSize(split_size);
                }
                node->output(i)->setType(
                    self_type->withSymbolicShapes(new_sizes));
              }
            }
          }
        }
        return;
      }
      case prim::Constant: {
        if (node->output()->type()->isSubtypeOf(TensorType::get())) {
          node->output()->inferTypeFrom(node->t(attr::value));
        }
        return;
      }
      case prim::unchecked_unwrap_optional: {
        // If we have specialized the optional type to the element type,
        // we want to pass it down. We write this as input.isSubtypeOf(output)
        // to be sure that we don't screw up nested optionals.
        if (node->input()->type()->isSubtypeOf(node->output()->type())) {
          node->output()->setType(node->input()->type());
        }
        return;
      }
      case prim::ConstantChunk: {
        Value* tensor = node->input();
        if (auto type = tensor->type()->cast<TensorType>()) {
          type = type->dimensionedOnly();
          for (Value* output : node->outputs()) {
            output->setType(type);
          }
        } else {
          setUnshapedType(node);
        }
        return;
      }
      case prim::grad: {
        auto tt = node->input()->type()->expect<TensorType>();
        // grad may be undefined
        // requires_grad may be required
        auto grad_type = TensorType::get()->withPossiblyUndefined();
        node->output()->setType(grad_type);
        return;
      }
      case prim::CallFunction:
      case prim::CallMethod:
      case prim::AutogradZero: {
        setUnshapedType(node);
        return;
      }
      case prim::GetAttr: {
        return;
      }
      case aten::_unwrap_optional: {
        // If we have specialized the optional type to the element type,
        // we want to pass it down. We write this as input.isSubtypeOf(output)
        // to be sure that we don't screw up nested optionals.
        if (node->input()->type()->isSubtypeOf(node->output()->type())) {
          node->output()->setType(node->input()->type());
        }
        return;
      }
      default:
        break; // fall-through
    }
    if (node->matches("aten::cat(Tensor[] tensors, int dim) -> Tensor") ||
        node->kind() == prim::FusedConcat) {
      return PropagateCatShape(node);
    }
#if PYTORCH_VERSION_GE(1, 14)
    if (auto maybe_complete_types =
            gatherTensorTypes(node, /*complete=*/true)) {
      if (PropagateCompleteShapeOnNode(
              node, insert_expands, std::move(*maybe_complete_types))) {
        return;
      }
    }
#endif

    SchemaSet scalar_schemas{
        "aten::gt.int(int a, int b) -> bool",
        "aten::lt.int(int a, int b) -> bool",
        "aten::ge.int(int a, int b) -> bool",
        "aten::le.int(int a, int b) -> bool",
        "aten::eq.int(int a, int b) -> bool",
        "aten::add.int(int a, int b) -> int",
        "aten::mul.int(int a, int b) -> int",
        "aten::sub.int(int a, int b) -> int",
        "aten::floordiv.int(int a, int b) -> int",
        "aten::gt.float(float a, float b) -> bool",
        "aten::lt.float(float a, float b) -> bool",
        "aten::ge.float(float a, float b) -> bool",
        "aten::le.float(float a, float b) -> bool",
        "aten::eq.float(float a, float b) -> bool",
        "aten::add.float(float a, float b) -> float",
        "aten::mul.float(float a, float b) -> float",
        "aten::sub.float(float a, float b) -> float",
    };

    SchemaSet no_shape_infer{
        "aten::split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]",
        "aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]",
        "aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]",
    };

    if (scalar_schemas.hasMember(*node) or no_shape_infer.hasMember(*node)) {
      // dtype already known or shape inference skipped
      return;
    }
    if (PropagateTensorShapeOnNode(node, insert_expands)) {
      return;
    }

    if (node->maybeSchema()) {
      LOG(WARNING) << "failed PropagateTensorShapeOnNode with schema: \n"
                   << node->schema() << "\n"
                   << *node;
    }

    // shape anaysis failed, erase traced shape only
    return setUnshapedType(node);
  }

  static c10::optional<size_t> determineListSize(Value* list) {
    AT_ASSERT(list->type()->cast<ListType>());
    if (auto ivalue = toIValue(list)) {
      if (ivalue->isList())
        return ivalue->toList().size();
    }
    if (auto shape = constant_as<c10::List<int64_t>>(list)) {
      return shape->size();
    }
    auto input_node = list->node();
    if (input_node->kind() == prim::ListConstruct) {
      return input_node->inputs().size();
    }
    if (input_node->kind() == aten::size) {
      if (auto type = input_node->input()->type()->cast<TensorType>()) {
        return type->dim();
      }
    }
    return c10::nullopt;
  }

  // is it ok to try to run the op
  // If an input is a constant, then we assume that the input is valid
  // and we can try to run it.
  // Otherwise:
  // Integral typed _inputs_ are often an indicator that we're indexing into
  // a tensor, so we should special-case these ops in the shape propagation.
  // Additionally, passing in a zero representative tensor into an integer
  // division op causes divide-by-zero errors
  // _Outputs_ must be tensors or primitives
  // We will call inferTypeFrom on the tensors, and ignore the primitives.
  // However, we allow primitive returns because we want to support mixed
  // primitive/tensor outputs.

  bool PropagateTensorShapeOnNode(Node* node, bool insert_expands) {
    static const auto broadcast =
        [](std::vector<TensorTypePtr>& tensor_types,
           c10::optional<at::ScalarType> t) -> TensorTypePtr {
      if (tensor_types.size() == 1) {
        return tensor_types[0]->dimensionedOnly()->withScalarType(t);
      }
      AT_ASSERT(!tensor_types.empty());
      auto any_type = tensor_types[0];
      auto max_dims = any_type->dim();
      for (auto& type : tensor_types) {
        if (!max_dims || !type->dim()) {
          max_dims = c10::nullopt;
        } else {
          max_dims = std::max(*max_dims, *type->dim());
        }
      }
      return TensorType::create(
          t,
          any_type->device(),
          max_dims,
          /*requires_grad=*/c10::nullopt);
    };

    using type_vec_t = std::vector<TensorTypePtr>;
    // Formula is expected to return a vector of length equal to the number of
    // tensor outputs of the node, or an empty vector which implies that it
    // failed to propagate.
    using formula_t = std::function<type_vec_t(Node*)>;
    static std::mutex shape_formulas_mutex;
    static std::vector<std::pair<SchemaSet, formula_t>> shape_formulas;
    struct register_formula_for {
      register_formula_for(SchemaSet operators, formula_t formula) {
        std::unique_lock<std::mutex> lock{shape_formulas_mutex};
        shape_formulas.emplace_back(std::move(operators), std::move(formula));
      }
    };

    static const register_formula_for first_input_type_formula{
        {
            "aten::alias(Tensor self) -> Tensor",
            "aten::erf(Tensor self) -> Tensor",
            "aten::erf_(Tensor self) -> Tensor",
            "aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor",
            "aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor",
            "aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)",
            "aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)",
            "aten::index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor",
            "aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
            "aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor",
#if PYTORCH_VERSION_GE(1, 13)
            "aten::select_scatter(Tensor self, Tensor src, int dim, int index) -> Tensor",
            "aten::slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor",
#elif PYTORCH_VERSION_GE(1, 11)
            "aten::slice_scatter(Tensor self, Tensor src, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor",
#endif
            "aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor",
            "aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
            "aten::relu(Tensor self) -> Tensor",
            "aten::relu_(Tensor self) -> Tensor",
#if PYTORCH_VERSION_GE(1, 12)
            "aten::gelu(Tensor self, *, str approximate='none') -> Tensor",
#else
            "aten::gelu(Tensor self) -> Tensor",
#endif
#if PYTORCH_VERSION_GE(1, 9)
            "aten::relu6(Tensor self) -> Tensor",
            "aten::relu6_(Tensor self) -> Tensor",
#endif
            "aten::acos(Tensor self) -> Tensor",
            "aten::bitwise_not(Tensor self) -> Tensor",
            "aten::neg(Tensor self) -> Tensor",
            "aten::sigmoid(Tensor self) -> Tensor",
            "aten::silu(Tensor self) -> Tensor",
            "aten::silu_(Tensor self) -> Tensor",
            "aten::logit(Tensor self, float? eps=None) -> Tensor",
            "aten::tanh(Tensor self) -> Tensor",
            "aten::asin(Tensor self) -> Tensor",
            "aten::atan(Tensor self) -> Tensor",
            "aten::ceil(Tensor self) -> Tensor",
            "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
            "aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)",
            "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor",
            "aten::celu(Tensor self, Scalar alpha) -> Tensor",
            "aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor",
            "aten::clamp_max(Tensor self, Scalar max) -> Tensor",
            "aten::clamp_min(Tensor self, Scalar min) -> Tensor",
            "aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor",
            "aten::cos(Tensor self) -> Tensor",
            "aten::cosh(Tensor self) -> Tensor",
            "aten::digamma(Tensor self) -> Tensor",
            "aten::dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::elu(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor",
            "aten::erfc(Tensor self) -> Tensor",
            "aten::erfinv(Tensor self) -> Tensor",
            "aten::exp(Tensor self) -> Tensor",
            "aten::expm1(Tensor self) -> Tensor",
            "aten::log(Tensor self) -> Tensor",
            "aten::log10(Tensor self) -> Tensor",
            "aten::log1p(Tensor self) -> Tensor",
            "aten::log2(Tensor self) -> Tensor",
            "aten::log_sigmoid(Tensor self) -> Tensor",
            "aten::floor(Tensor self) -> Tensor",
            "aten::frac(Tensor self) -> Tensor",
            "aten::flip(Tensor self, int[] dims) -> Tensor",
            "aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::feature_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::hardshrink(Tensor self, Scalar lambd) -> Tensor",
            "aten::hardtanh(Tensor self, Scalar min_val, Scalar max_val) -> Tensor",
            "aten::inverse(Tensor self) -> Tensor",
            "aten::group_norm(Tensor input, int num_groups, Tensor? weight, Tensor? bias, float eps, bool cudnn_enabled) -> Tensor",
            "aten::leaky_relu(Tensor self, Scalar negative_slope) -> Tensor",
            "aten::leaky_relu_(Tensor self, Scalar negative_slope) -> Tensor",
            "aten::lgamma(Tensor self) -> Tensor",
            "aten::mvlgamma(Tensor self, int p) -> Tensor",
            "aten::normal(float mean, Tensor std, *, Generator? generator) -> Tensor",
            "aten::normal(Tensor mean, float std, *, Generator? generator) -> Tensor",
#if PYTORCH_VERSION_GE(1, 12)
            "aten::pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)",
            "aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate='none') -> Tensor",
            "aten::native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor",
            "aten::fill.Scalar(Tensor self, Scalar value) -> Tensor",
            "aten::fill.Tensor(Tensor self, Tensor value) -> Tensor",
#endif
            "aten::pinverse(Tensor self, float rcond) -> Tensor",
            "aten::reciprocal(Tensor self) -> Tensor",
            "aten::relu(Tensor self) -> Tensor",
            "aten::relu_(Tensor self) -> Tensor",
#if PYTORCH_VERSION_GE(1, 9)
            "aten::relu6(Tensor self) -> Tensor",
            "aten::relu6_(Tensor self) -> Tensor",
#endif
            "aten::round(Tensor self) -> Tensor",
            "aten::rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
            "aten::rsqrt(Tensor self) -> Tensor",
            "aten::selu(Tensor self) -> Tensor",
            "aten::selu_(Tensor self) -> Tensor",
            "aten::sigmoid(Tensor self) -> Tensor",
            "aten::sign(Tensor self) -> Tensor",
            "aten::sin(Tensor self) -> Tensor",
            "aten::sinh(Tensor self) -> Tensor",
            "aten::softplus(Tensor self, Scalar beta, Scalar threshold) -> Tensor",
            "aten::softshrink(Tensor self, Scalar lambd) -> Tensor",
            "aten::sqrt(Tensor self) -> Tensor",
            "aten::tan(Tensor self) -> Tensor",
            "aten::tanh(Tensor self) -> Tensor",
            "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor",
            "aten::tril(Tensor self, int diagonal) -> Tensor",
            "aten::triu(Tensor self, int diagonal) -> Tensor",
            "aten::trunc(Tensor self) -> Tensor",
            "aten::rot90(Tensor self, int k, int[] dims) -> Tensor",
            "aten::alias(Tensor self) -> Tensor",
            "aten::zero_(Tensor self) -> Tensor",
            "aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor",
            "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)",
#ifdef TORCH_BLADE_BUILD_QUANTIZATION
            "torch_blade::fake_quant(Tensor _0, Tensor _1, Tensor _2, int _3, int _4, int _5, int[] _6, bool _7, bool _8, bool _9, bool _10) -> Tensor",
            "torch_blade::placeholder(Tensor _0) -> (Tensor _0)",
#endif
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {type};
          }
          return {};
        }};

    // Requirements:
    //   scalar type    : preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for simple_unary_ops{
        {
            "aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
#if PYTORCH_VERSION_GE(1, 14)
            "aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, SymInt length) -> Tensor(a)",
#else
            "aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)",
#endif
            "aten::t(Tensor self) -> Tensor",
            "aten::transpose(Tensor self, int dim0, int dim1) -> Tensor",
#if PYTORCH_VERSION_GE(1, 11)
            "aten::constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor",
            "aten::pad(Tensor self, SymInt[] pad, str mode, float? value=None) -> Tensor",
#else
            "aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor",
#endif
        },
        [](Node* node) -> type_vec_t {
          if (auto input_type = node->input(0)->type()->cast<TensorType>()) {
            return type_vec_t{input_type->dimensionedOnly()};
          }
          return type_vec_t{};
        }};

    static const register_formula_for permute_op{
        {
            "aten::permute(Tensor self, int[] dims) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto input_type = node->input(0)->type()->cast<TensorType>()) {
            if (auto dims = constant_as<c10::List<int64_t>>(
                    node->namedInput(attr::dims))) {
              std::vector<ShapeSymbol> new_dims(dims->size());
              for (size_t i = 0; i < dims->size(); ++i) {
                auto dim_size = getSymDimSize(input_type, dims.value()[i]);
                new_dims[i] = dim_size;
              }
              return type_vec_t{input_type->withSymbolicShapes(new_dims)};
            }
            return type_vec_t{input_type->dimensionedOnly()};
          }
          return type_vec_t{};
        }};
    // Requirements:
    //   dims           : preserved
    //   scalar type    : preserved, except complex maps to float
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for simple_unary_ops_complex_to_float{
        {
            "aten::abs(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          auto input_type = node->input(0)->type()->cast<TensorType>();

#if PYTORCH_VERSION_GE(1, 12)
          // Maps complex -> float
          if (input_type->scalarType()) {
            const auto scalar_type = *(input_type->scalarType());
            if (isComplexType(scalar_type)) {
              const auto out_type = c10::toRealValueType(scalar_type);
              return type_vec_t{
                  input_type->dimensionedOnly()->withScalarType(out_type)};
            }
          }
#endif
          return input_type ? type_vec_t{input_type->dimensionedOnly()}
                            : type_vec_t{};
        }};

    // Requirements:
    //   dims           : preserved
    //   scalar type    : preserved
    //   device         : preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for inplace_ops_arithmetic{
        {
            // Tensor-Tensor operators
            "aten::add_(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::add_(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::sub_(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::sub_(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::mul_(Tensor self, Tensor other) -> Tensor",
            "aten::mul_(Tensor self, Scalar other) -> Tensor",
            "aten::div_(Tensor self, Tensor other) -> Tensor",
            "aten::div_(Tensor self, Scalar other) -> Tensor",
#if PYTORCH_VERSION_GE(1, 9)
            "aten::div_.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor",
#endif // PYTORCH_VERSION_GE(1, 9)
            "aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
            "aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
            "aten::add_inplace(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::sub_inplace(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::mul_inplace(Tensor self, Tensor other) -> Tensor",
            "aten::div_inplace(Tensor self, Tensor other) -> Tensor",
            "aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> (Tensor(a!))",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {type};
          }
          return {};
        }};

    // Requirements:
    //   dims           : broadcast all tensor args
    //   scalar type    : promoted from input dtypes
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for broadcasting_ops_arithmetic{
        {
            // Tensor-Tensor operators
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::rsub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::mul(Tensor self, Tensor other) -> Tensor",
            "aten::div(Tensor self, Tensor other) -> Tensor",
#if PYTORCH_VERSION_GE(1, 9)
            "aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor",
#endif // PYTORCH_VERSION_GE(1, 9)
            "aten::floor_divide(Tensor self, Tensor other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            AT_ASSERT(maybe_tensor_types->size() >= 2);
            auto dtype = getPromotedTypeForArithmeticOp(node);
#if PYTORCH_VERSION_GE(1, 9)
            if (node->matches(
                    "aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor")) {
              return {broadcast(*maybe_tensor_types, dtype)};
            }
#endif // PYTORCH_VERSION_GE(1, 9)

            if ((node->kind() == aten::div || node->kind() == aten::div_) &&
                dtype.has_value() &&
                c10::isIntegralType(dtype.value(), false)) {
              dtype = at::typeMetaToScalarType(c10::get_default_dtype());
            }
            return {broadcast(*maybe_tensor_types, dtype)};
          }
          return {};
        }};

    // Requirements:
    //   dims           : broadcast all tensor args
    //   scalar type    : always matching and preserved
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for broadcasting_ops{
        {
            "aten::fmod(Tensor self, Tensor other) -> Tensor",
            "aten::remainder(Tensor self, Tensor other) -> Tensor",
            "aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor",
            "aten::lerp(Tensor self, Tensor end, Tensor weight) -> Tensor",
            "aten::max(Tensor self, Tensor other) -> Tensor",
            "aten::min(Tensor self, Tensor other) -> Tensor",
            "aten::__and__(Tensor self, Tensor other) -> Tensor",
            "aten::__or__(Tensor self, Tensor other) -> Tensor",
            "aten::__xor__(Tensor self, Tensor other) -> Tensor",
            "aten::__lshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__rshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__iand__(Tensor self, Tensor other) -> Tensor",
            "aten::__ior__(Tensor self, Tensor other) -> Tensor",
            "aten::__ixor__(Tensor self, Tensor other) -> Tensor",
            "aten::__ilshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__irshift__(Tensor self, Tensor other) -> Tensor",

            // Ops with Tensor-Tensor overloads only
            "aten::atan2(Tensor self, Tensor other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            AT_ASSERT(maybe_tensor_types->size() >= 2);
            auto first_scalar_type = (*maybe_tensor_types)[0]->scalarType();
            auto second_scalar_type = (*maybe_tensor_types)[1]->scalarType();
            if (!first_scalar_type || !second_scalar_type) {
              return {};
            }
            size_t arg_for_type = 0;
            if (c10::promoteTypes(*first_scalar_type, *second_scalar_type) !=
                first_scalar_type) {
              arg_for_type = 1;
            }
            auto t = (*maybe_tensor_types)[arg_for_type]->scalarType();
            return {broadcast(*maybe_tensor_types, *t)};
          }
          return {};
        }};

    static const register_formula_for fused_accum_binary_ops{
        {
            // Non-binary ops
            "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value) -> Tensor",
            "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            auto dtype = (*maybe_tensor_types)[0]->scalarType();
            if (!dtype) {
              return {};
            }
            return {broadcast(*maybe_tensor_types, *dtype)};
          }
          return {};
        }};

    static const register_formula_for broadcasting_scalar_tensor_ops_arithmetic{
        {
            "aten::pow(Scalar self, Tensor exponent) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          std::cout << "infer aten::pow" << std::endl;
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            auto dtype = (*maybe_tensor_types)[0]->scalarType();
            if (!dtype)
              return {};
            return {broadcast(*maybe_tensor_types, dtype)};
          }
          return {};
        }};
    static const register_formula_for broadcasting_tensor_scalar_ops_arithmetic{
        {
            // Tensor-Scalar operators
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
            "aten::mul(Tensor self, Scalar other) -> Tensor",
            "aten::div(Tensor self, Scalar other) -> Tensor",
            "aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            auto first_scalar_type = (*maybe_tensor_types)[0]->scalarType();
            auto second_scalar_type =
                tryScalarTypeFromJitType(*node->inputs()[1]->type());
            if (!first_scalar_type || !second_scalar_type) {
              return {};
            }
            if ((isIntegralType(*first_scalar_type, false) &&
                 isFloatingType(*second_scalar_type)) ||
                (isIntegralType(*first_scalar_type, false) &&
                 isIntegralType(*second_scalar_type, false) &&
                 (node->kind() == aten::div || node->kind() == aten::div_))) {
              auto default_dtype =
                  at::typeMetaToScalarType(c10::get_default_dtype());
              return {broadcast(*maybe_tensor_types, default_dtype)};
            }
            if (c10::ScalarType::Bool == *first_scalar_type &&
                c10::ScalarType::Bool != *second_scalar_type) {
              auto result_type =
                  c10::promoteTypes(*first_scalar_type, *second_scalar_type);
              return {broadcast(*maybe_tensor_types, result_type)};
            }
            return {broadcast(*maybe_tensor_types, first_scalar_type)};
          }
          return {};
        }};

    // NB: we always take the scalar type of the Tensor
    static const register_formula_for broadcasting_tensor_scalar_ops{
        {

            "aten::fmod(Tensor self, Scalar other) -> Tensor",
            "aten::remainder(Tensor self, Scalar other) -> Tensor",
            "aten::__and__(Tensor self, Scalar other) -> Tensor",
            "aten::__or__(Tensor self, Scalar other) -> Tensor",
            "aten::__xor__(Tensor self, Scalar other) -> Tensor",
            "aten::__lshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__rshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__iand__(Tensor self, Scalar other) -> Tensor",
            "aten::__ior__(Tensor self, Scalar other) -> Tensor",
            "aten::__ixor__(Tensor self, Scalar other) -> Tensor",
            "aten::__ilshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__irshift__(Tensor self, Scalar other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            return {broadcast(
                *maybe_tensor_types, (*maybe_tensor_types)[0]->scalarType())};
          }
          return {};
        }};

    // aten::where is special in that its return type is the second argument's
    // (self) type rather than the that of condition
    static const register_formula_for where_op{
        {
            "aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor",
            "aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> (Tensor)",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            return {broadcast(
                *maybe_tensor_types, (*maybe_tensor_types)[1]->scalarType())};
          }
          return {};
        }};

    static const auto any_tensor_type = [](Node* node) -> TensorTypePtr {
      for (Value* input : node->inputs()) {
        if (auto type = input->type()->cast<TensorType>()) {
          if (type->dim().has_value()) {
            return type;
          }
        }
      }
      return nullptr;
    };

    // Requirements:
    //   dims           : always matching and preserved
    //   scalar type    : always matching and preserved
    //   device         : always matching and preserved
    //   tensor inputs  : 2
    //   tensor outputs : 1
    static const register_formula_for binary_ops_strict_match{
        {
            "aten::normal(Tensor mean, Tensor std, *, Generator? generator) -> Tensor",
            "aten::mm(Tensor self, Tensor mat2) -> Tensor",
            "aten::bmm(Tensor self, Tensor mat2) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          auto t = node->input(0)->type()->cast<TensorType>();
          if (t) {
            return {t->dimensionedOnly()};
          } else {
            return {};
          }
        }};

    // Requirements:
    //   dims           : all tensor args are broadcast
    //   scalar type    : byte/uint8
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for comparison_ops{
        {
            "aten::lt(Tensor self, Tensor other) -> Tensor",
            "aten::le(Tensor self, Tensor other) -> Tensor",
            "aten::gt(Tensor self, Tensor other) -> Tensor",
            "aten::ge(Tensor self, Tensor other) -> Tensor",
            "aten::eq(Tensor self, Tensor other) -> Tensor",
            "aten::ne(Tensor self, Tensor other) -> Tensor",
            "aten::lt(Tensor self, Scalar other) -> Tensor",
            "aten::le(Tensor self, Scalar other) -> Tensor",
            "aten::gt(Tensor self, Scalar other) -> Tensor",
            "aten::ge(Tensor self, Scalar other) -> Tensor",
            "aten::eq(Tensor self, Scalar other) -> Tensor",
            "aten::ne(Tensor self, Scalar other) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            return {broadcast(*maybe_tensor_types, at::kBool)};
          }
          return {};
        }};

    static const register_formula_for nn_ops_first_input_formula{
        *nn_ops_first_input_preserving(), [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {type->dimensionedOnly()};
          }
          return {};
        }};

    // Requirements:
    //   dims           : preserved
    //   scalar type    : dtype
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for aten_to_dtype{
        {"aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
         "aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)",
#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION >= 8
         "aten::to.dtype_layout(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor"
#endif
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            auto ret = type;
            auto dtype = getScalarTypeFromValue(node->namedInput(attr::dtype));
            if (dtype) {
              return {ret->withScalarType(dtype)};
            } else {
              return {ret};
            }
          }
          return {};
        }};

    // Requirements:
    //   device         : Device
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for aten_to_device{
        {"aten::to.device(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor",
         "aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)"},
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            auto device = getDeviceFromValue(node->namedInput(attr::device));
            if (type->dim()) {
              auto scalarType =
                  getScalarTypeFromValue(node->namedInput(attr::dtype));
              if (!scalarType) {
                scalarType = type->scalarType();
              }
              return {TensorType::create(
                          scalarType,
                          device,
                          type->dim(),
                          /*requires_grad=*/c10::nullopt)
                          ->withSymbolicShapes(type->symbolic_sizes())};
            }
          }
          return {};
        }};

    // Requirements:
    //   device         : cuda
    //   tensor inputs  : 1
    //   tensor outputs : 1
    static const register_formula_for aten_cuda{
        {"aten::cuda(Tensor(a) self) -> Tensor(a|b)"},
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            if (type->dim()) {
              return {TensorType::create(
                          type->scalarType(),
                          at::kCUDA,
                          type->dim(),
                          /*requires_grad=*/c10::nullopt)
                          ->withSymbolicShapes(type->symbolic_sizes())};
            }
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for all_reduce_ops{
        {
            "aten::det(Tensor self) -> Tensor",
            "aten::logdet(Tensor self) -> Tensor",
            "aten::max(Tensor self) -> Tensor",
            "aten::min(Tensor self) -> Tensor",
            "aten::median(Tensor self) -> Tensor",
#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION > 7
            "aten::nanmedian(Tensor self) -> Tensor",
#endif
            "aten::norm(Tensor self, Scalar p) -> Tensor",
            "aten::std(Tensor self, bool unbiased) -> Tensor",
            "aten::trace(Tensor self) -> Tensor",
            "aten::var(Tensor self, bool unbiased) -> Tensor",
            "aten::all(Tensor self) -> Tensor",
            "aten::any(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {type->withDim(0)};
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : dtype if specified, else preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for reduce_ops_with_opt_dtype{
        {"aten::mean(Tensor self, *, int? dtype) -> Tensor"},
        [](Node* node) -> type_vec_t {
          at::optional<IValue> maybe_dtype_option = node->get(attr::dtype);
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            auto ret = type->withDim(0);
            auto dtype = getScalarTypeFromValue(node->namedInput(attr::dtype));
            if (dtype) {
              return {ret->withScalarType(dtype)};
            } else {
              return {ret};
            }
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : dtype if specified, else preserved if floating point,
    //   otherwise long/int64 device         : preserved tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for
        all_reduce_ops_with_integer_upcast_and_dtype{
            {
                "aten::sum(Tensor self, *, int? dtype) -> Tensor",
                "aten::prod(Tensor self, *, int? dtype) -> Tensor",
            },
            [](Node* node) -> type_vec_t {
              if (auto type = node->input(0)->type()->cast<TensorType>()) {
                type = type->withDim(0);
                auto dtype =
                    getScalarTypeFromValue(node->namedInput(attr::dtype));
                if (dtype) {
                  return {type->withScalarType(dtype)};
                }
                if (type->scalarType()) {
                  return {
                      at::isFloatingType(*type->scalarType())
                          ? type
                          : type->withScalarType(at::kLong)};
                } else {
                  return {type};
                }
              }
              return {};
            }};

    static const auto reduce_op_handler =
        [](Node* node,
           int64_t num_reduced_dim = 0,
           bool upcast_integer = false,
           c10::optional<IValue> opt_dtype = c10::nullopt,
           c10::optional<c10::List<int64_t>> opt_dims =
               c10::nullopt) -> type_vec_t {
      if (auto type = node->input(0)->type()->cast<TensorType>()) {
        if (!type->scalarType() || !type->dim()) {
          return {};
        }
        if (opt_dtype && !opt_dtype->isNone()) {
          type = type->withScalarType(opt_dtype->toScalarType());
        } else if (upcast_integer && !at::isFloatingType(*type->scalarType())) {
          type = type->withScalarType(at::kLong);
        }
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        if (*type->dim() >= num_reduced_dim && num_reduced_dim >= 0) {
          return {type->withDim(*type->dim() - num_reduced_dim)};
        } else {
          auto sizesOptional = type->symbolic_sizes().sizes();
          if (opt_dims && sizesOptional) {
            std::vector<ShapeSymbol> new_sizes = sizesOptional.value();
            for (ssize_t k = 0; k < opt_dims->size(); ++k) {
              int64_t dim = opt_dims.value()[k];
              if (dim < 0)
                dim += new_sizes.size();
              new_sizes[dim] = ShapeSymbol::fromStaticSize(1);
            }
            return {type->withSymbolicShapes(new_sizes)};
          }
          return {type->dimensionedOnly()};
        }
      }
      return {};
    };

    static const auto multidim_reduce_with_keepdim =
        [](Node* node,
           int64_t num_reduced_dim,
           bool upcast_integer,
           c10::optional<IValue> opt_dtype = c10::nullopt,
           c10::optional<c10::List<int64_t>> opt_dims =
               c10::nullopt) -> type_vec_t {
      auto maybe_keepdim = node->get<bool>(attr::keepdim);
      if (!maybe_keepdim)
        return {};
      return reduce_op_handler(
          node,
          *maybe_keepdim ? 0 : num_reduced_dim,
          upcast_integer,
          opt_dtype,
          opt_dims);
    };

    // Requirements:
    //   dims           : 0 if keepdim, otherwise n(dims) smaller
    //   scalar type    : dtype if given
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    //   - Has a bool keepdim argument
    static const register_formula_for multidim_reduce_ops_with_integer_upcast_and_dtype{
        {
#if PYTORCH_VERSION_GE(1, 14)
            "aten::sum.dim_IntList(Tensor self, int[]? dim, bool keepdim, *, int? dtype) -> Tensor",
            "aten::mean.dim(Tensor self, int[]? dim, bool keepdim, *, int? dtype) -> Tensor",
            "aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor",
#else
            "aten::sum(Tensor self, int[] dim, bool keepdim, *, int? dtype) -> Tensor",
            "aten::mean(Tensor self, int[] dim, bool keepdim, *, int? dtype) -> Tensor",
#endif
#if PYTORCH_VERSION_GE(2, 1)
            "aten::var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor",
#elif PYTORCH_VERSION_GE(1, 14)
            "aten::var.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor",
#endif
        },
        [](Node* node) -> type_vec_t {
          auto list = node->namedInput(attr::dim);
          auto num_reduced_dim = determineListSize(list);
          if (!num_reduced_dim) {
            return {};
          }

          at::optional<IValue> opt_dtype;
#if PYTORCH_VERSION_GE(2, 1)
          if (!(node->matches(
                    "aten::var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor") ||
                node->matches(
                    "aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"))) {
            opt_dtype = node->get(attr::dtype);
          }
#elif PYTORCH_VERSION_GE(1, 14)
          if (!(node->matches(
                    "aten::var.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor") ||
                node->matches(
                    "aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"))) {
            opt_dtype = node->get(attr::dtype);
          }
#else
          opt_dtype = node->get(attr::dtype);
#endif
          auto dims = constant_as<c10::List<int64_t>>(list);
          return multidim_reduce_with_keepdim(
              node,
              /*num_reduced_dim=*/*num_reduced_dim,
              /*upcast_integer=*/true,
              opt_dtype,
              dims);
        }};

    // Requirements:
    //   dims           : 0 if dim is None, otherwise preserved if keepdim ==
    //   false or 1 smaller otherwise scalar type    : preserved device :
    //   preserved tensor inputs  : 1 tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    //   - Has a bool keepdim argument
    static const register_formula_for argminmax{
        {
            "aten::argmax(Tensor self, int? dim, bool keepdim) -> Tensor",
            "aten::argmin(Tensor self, int? dim, bool keepdim) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            if (node->input(1)->type()->kind() == c10::TypeKind::NoneType) {
              return {type->withDim(0)};
            } else {
              return multidim_reduce_with_keepdim(
                  node, /*num_reduced_dim=*/1, /*upcast_integer=*/false);
            }
          }
          return {};
        }};

    // Requirements:
    //   dims           : preserved if keepdim == false, 1 smaller otherwise
    //   scalar type    : preserved for first output, byte/uint8 for second
    //   output if exists device         : preserved tensor inputs  : 1 tensor
    //   outputs : 1 or 2
    // Additionally:
    //   - First input should be the only tensor input
    //   - Has a bool keepdim argument
    static const register_formula_for dim_reduce_ops{
        {
            "aten::all(Tensor self, int dim, bool keepdim) -> Tensor",
            "aten::any(Tensor self, int dim, bool keepdim) -> Tensor",

            // Ops returning indices as second output
            "aten::kthvalue(Tensor self, int k, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::max(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::min(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::median(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION > 7
            "aten::nanmedian(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
#endif
            "aten::mode(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
        },
        [](Node* node) -> type_vec_t {
          // NB: Note that while this function is generally meant to be used
          // with ops that have a single output, we will fix up its return right
          // below.
          auto output_types = multidim_reduce_with_keepdim(
              node, /*num_reduced_dim=*/1, /*upcast_integer=*/false);
          if (!output_types.empty() && node->outputs().size() == 2) {
            output_types.push_back(
                output_types.back()->withScalarType(at::kLong));
          }
          return output_types;
        }};

    static const register_formula_for dim_squeeze_ops{
        {
            "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)",
        },
        [](Node* node) -> type_vec_t {
          auto dimOptional = node->get<int64_t>(attr::dim);
          if (dimOptional) {
            if (auto type = node->input(0)->type()->cast<TensorType>()) {
              auto sym_dim_size = getSymDimSize(type, dimOptional.value());
              if (sym_dim_size.is_static() and
                  sym_dim_size.static_size() != 1) {
                return {type};
              }
            }
            // TODO(tanyo): the analysis can't handle dynamic rank correctly,
            // it presumed squeeze.dim always remove 1 dimension
            c10::List<int64_t> dims{dimOptional.value()};
            return reduce_op_handler(
                node,
                /*num_reduce_dim=*/1,
                /*integer_upcast=*/false,
                c10::nullopt,
                dims);
          } else {
            return reduce_op_handler(node, /*num_reduce_dim=*/1);
          }
        }};

    static const register_formula_for dims_squeeze_ops{
        {
            "aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)",
        },
        [](Node* node) -> type_vec_t {
          if (auto squeeze_dims = constant_as<c10::List<int64_t>>(
                  node->namedInput(attr::dim))) {
            if (auto type = node->input(0)->type()->cast<TensorType>()) {
              if (type->sizes().concrete_sizes().has_value()) {
                auto dims = type->sizes().concrete_sizes().value();
                std::vector<int64_t> new_dims;
                for (int i = 0; i < dims.size(); ++i) {
                  if (dims[i] != 1 ||
                      std::find(
                          squeeze_dims->vec().begin(),
                          squeeze_dims->vec().end(),
                          i) == squeeze_dims->vec().end()) {
                    new_dims.push_back(dims[i]);
                  }
                }
                return {type->withSizes(new_dims)};
              }
              return reduce_op_handler(
                  node,
                  squeeze_dims.value().size(),
                  false,
                  c10::nullopt,
                  squeeze_dims);
            }
          }
          return {};
        }};

    // Requirements:
    //   dims           : preserved if keepdim == false, 1 smaller otherwise
    //   scalar type    : dtype if specified. preserved if floating point,
    //   otherwise long/int64 device         : preserved tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    //   - has a bool keepdim argument
    static const register_formula_for dim_reduce_ops_with_integer_upcast{
        {
            "aten::prod(Tensor self, int dim, bool keepdim, *, int? dtype) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          auto maybe_keepdim = node->get<bool>(attr::keepdim);
          at::optional<IValue> opt_dtype = node->get(attr::dtype);
          return reduce_op_handler(
              node,
              /*num_reduce_dim=*/*maybe_keepdim ? 0 : 1,
              /*integer_upcast=*/true,
              opt_dtype);
        }};

    // Requirements:
    //   dims           : preserved
    //   scalar type    : dtype if specified, preserved if floating point,
    //    otherwise long/int64
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for dim_reduce_ops_dtype{
        {"aten::cumprod(Tensor self, int dim, *, int? dtype) -> Tensor",
         "aten::cumsum(Tensor self, int dim, *, int? dtype) -> Tensor",
         "aten::log_softmax(Tensor self, int dim, int? dtype) -> Tensor"},
        [](Node* node) -> type_vec_t {
          at::optional<IValue> opt_dtype = node->get(attr::dtype);
          return reduce_op_handler(
              node, /*num_reduce_dim=*/0, /*integer_upcast=*/true, opt_dtype);
        }};

    // Requirements:
    //   dims           : preserved
    //   scalar type    : half to float if specified, otherwise preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    static const register_formula_for register__softmax{
        {"aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor",
         "aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"},
        [](Node* node) -> type_vec_t {
          bool half_to_float =
              node->get<bool>(Symbol::attr("half_to_float")).value();
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            if (half_to_float) {
              return {type->withScalarType(at::kFloat)};
            }
            return {type};
          }
          return {};
        }};

    // Requirements:
    //   dims           : preserved
    //   scalar type    : half to float if specified, otherwise preserved
    //   device         : preserved
    //   tensor inputs  : 1
#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION >= 12
    static const register_formula_for register__softmax_backward{
        {"aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor",
         "aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor"},
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            auto scalar_type = type->scalarType();
            auto input_dtype = node->get(attr::input_dtype)->toScalarType();
            auto half_to_float = scalar_type != input_dtype;
            if (half_to_float && scalar_type == at::kFloat &&
                input_dtype == at::kHalf) {
              return {type->withScalarType(at::kHalf)};
            }
            return {type};
          }
          return {};
        }};
#endif
    static const register_formula_for register__nll_loss_backward{
        {"aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> (Tensor)"},
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(1)->type()->cast<TensorType>()) {
            return {type};
          }
          return {};
        }};

    // Requirements:
    //   dims           : preserved
    //   scalar type    : dtype if specified, otherwise preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    static const register_formula_for register_softmax{
        {
            "aten::softmax(Tensor self, int dim, int? dtype) -> Tensor",
            "aten::log_softmax.int(Tensor self, int dim, int? dtype=None) -> (Tensor)",
        },
        [](Node* node) -> type_vec_t {
          at::optional<IValue> opt_dtype = node->get(attr::dtype);
          return reduce_op_handler(
              node, /*num_reduced_dim=*/0, /*upcast_integer=*/false, opt_dtype);
        }};

    static const auto factory_with_ndim = [](Node* node,
                                             int dim) -> type_vec_t {
      auto device = getDeviceFromValue(node->namedInput(attr::device));

      auto dtype = at::kFloat;
      if (auto scalarType =
              getScalarTypeFromValue(node->namedInput(attr::dtype))) {
        dtype = *scalarType;
      }
      return {TensorType::create(
          dtype, device, dim, /*requires_grad=*/c10::nullopt)};
    };

    static const auto factory_like_with_ndim = [](Node* node,
                                                  int dim,
                                                  at::ScalarType default_dtype =
                                                      at::kHalf) -> type_vec_t {
      at::ScalarType dtype = default_dtype;
      if (auto scalarType =
              getScalarTypeFromValue(node->namedInput(attr::dtype))) {
        dtype = *scalarType;
      }

      auto device = getDeviceFromValue(node->namedInput(attr::device));
      return {TensorType::create(
          dtype, device, dim, /*requires_grad=*/c10::nullopt)};
    };

    // Requirements:
    //   dims           : preserved
    //   scalar type    : equal to value of dtype
    //   device         : equal to value of device
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - has ScalarType dtype, Layeout layout and Device device arguments
    static const register_formula_for like_factories_with_options{
        {
            "aten::empty_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::ones_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::rand_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::randint_like(Tensor self, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::randint_like(Tensor self, int low, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::randn_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::zeros_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
#if PYTORCH_VERSION_GE(1, 12)
            "aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor",
#elif PYTORCH_VERSION_GE(1, 10)
            "aten::_to_copy(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor",
#endif
        },
        [](Node* node) -> type_vec_t {
          if (auto type =
                  node->namedInput(attr::self)->type()->cast<TensorType>()) {
            if (type->dim()) {
              type_vec_t type_v;
              if (auto input_scalar_type = type->scalarType()) {
                type_v = factory_like_with_ndim(
                    node, (int)*type->dim(), input_scalar_type.value());
              } else {
                type_v = factory_like_with_ndim(node, (int)*type->dim());
              }
              if (type->isComplete()) {
                type_v[0] = type_v[0]->withSizes(
                    type->sizes().concrete_sizes().value());
              }
              return type_v;
            }
          }
          return {};
        }};

    // Requirements:
    //   dims           : equal to number of elements in size
    //   scalar type    : equal to value of dtype
    //   device         : equal to value of device
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - has int[] size, ScalarType dtype, Layeout layout and Device device
    //   arguments
    static const register_formula_for new_size_factories_with_options{
        {
#if PYTORCH_VERSION_GE(1, 13)
            "aten::new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
#else
            "aten::new_zeros(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
#endif
        },
        [](Node* node) -> type_vec_t {
          if (auto type =
                  node->namedInput(attr::self)->type()->cast<TensorType>()) {
            auto dim = determineListSize(node->namedInput(attr::size));
            if (dim)
              return factory_like_with_ndim(node, *dim);
          }
          return {};
        }};

    // Requirements:
    //   dims           : equal to number of elements in size
    //   scalar type    : equal to value of dtype
    //   device         : equal to value of device
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - has int[] size, ScalarType dtype, Layeout layout and Device device
    //   arguments
    static const register_formula_for size_factories_with_options{
        {
            "aten::empty(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory, MemoryFormat? memory_format=contiguous_format) -> Tensor",
            "aten::full(int[] size, Scalar fill_value, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::ones(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::randn(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::zeros(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::randint(int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::randint(int low, int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
        },
        [&](Node* node) -> type_vec_t {
          if (auto list_size =
                  determineListSize(node->namedInput(attr::size))) {
            auto ret = factory_with_ndim(node, *list_size);
            if (node->kind() == aten::full) {
              at::optional<IValue> maybe_dtype_option = node->get(attr::dtype);
              if (maybe_dtype_option && maybe_dtype_option->isInt()) {
                return ret;
              }
              at::ScalarType dtype =
                  *tryScalarTypeFromJitType(*node->input(1)->type());
              return {ret[0]->withScalarType(dtype)};
            }
            return ret;
          }
          return {};
        }};

    // Requirements:
    //   dims           : has rank 1
    //   scalar type    : equal to value of dtype
    //   device         : equal to value of device
    //   tensor outputs : 1
    // Additionally:
    //   - has ScalarType dtype, Layout layout and Device device
    //   arguments
    static const register_formula_for rank1_factories_with_options{
        {
            "aten::arange(Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",
            "aten::arange.start(Scalar start, Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",
            "aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",
        },
        [&](Node* node) -> type_vec_t {
          at::ScalarType dtype =
              *tryScalarTypeFromJitType(*node->input(0)->type());
          at::optional<IValue> maybe_dtype_option = node->get(attr::dtype);
          if (maybe_dtype_option && maybe_dtype_option->isInt()) {
            dtype = maybe_dtype_option->toScalarType();
          }

          auto device = getDeviceFromValue(node->namedInput(attr::device));
          return {TensorType::create(
              dtype, device, /*dim=*/1, /*requires_grad=*/c10::nullopt)};
        }};

    static const auto get_cast_scalar_type = [](Node* node) -> at::ScalarType {
      switch (node->kind()) {
        case aten::_cast_Byte:
          return at::kByte;
        case aten::_cast_Char:
          return at::kChar;
        case aten::_cast_Double:
          return at::kDouble;
        case aten::_cast_Float:
          return at::kFloat;
        case aten::_cast_Half:
          return at::kHalf;
        case aten::_cast_Int:
          return at::kInt;
        case aten::_cast_Long:
          return at::kLong;
        case aten::_cast_Short:
          return at::kShort;
        default:
          AT_ASSERTM(
              false,
              "unknown node kind in get_cast_scalar_type: ",
              node->kind().toQualString());
      }
    };
    static const register_formula_for cast_ops{
        {
            "aten::_cast_Byte(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Char(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Double(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Float(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Half(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Int(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Long(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Short(Tensor self, bool non_blocking) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type =
                  node->namedInput(attr::self)->type()->cast<TensorType>()) {
            return {type->withScalarType(get_cast_scalar_type(node))};
          }
          return {};
        }};

#if PYTORCH_VERSION_GE(1, 12)
    // refer to
    // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/codegen/cuda/type_inference.cpp#L494
    static const register_formula_for autocast_ops{
        {
            "aten::_autocast_to_reduced_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) -> Tensor(a)",
            "aten::_autocast_to_full_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled) -> Tensor(a)",
        },
        [](Node* node) -> type_vec_t {
          const auto in_type = node->input(0)->type()->cast<TensorType>();
          const auto in_scalar_type = in_type->scalarType();

          // reduced_precision
          if (node->hasNamedInput("cuda_dtype")) {
            if (in_scalar_type == at::ScalarType::Float) {
              bool cuda_enabled = node->get<bool>(attr::cuda_enabled).value();
              return {in_type->withScalarType(
                  node->get(cuda_enabled ? attr::cuda_dtype : attr::cpu_dtype)
                      ->toScalarType())};
            }
          }
          // full_precision
          else {
            if (in_scalar_type == at::ScalarType::Half ||
                in_scalar_type == at::ScalarType::BFloat16) {
              return {in_type->withScalarType(at::ScalarType::Float)};
            }
          }
          return {in_type};
        }};
#endif
    static const register_formula_for reshape_op{
        {
            "aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)",
        },
        [&](Node* node) -> type_vec_t {
          auto self = node->namedInput(attr::self);
          auto shape = node->namedInput(attr::shape);
          if (auto self_type = self->type()->cast<TensorType>()) {
            if (auto dims = constant_as<c10::List<int64_t>>(shape)) {
              std::vector<int64_t> newDims;
              for (size_t i = 0; i < dims->size(); ++i)
                newDims.push_back((*dims)[i]);
              return {self_type->withSizes(newDims)};
            }
            if (auto list_size = determineListSize(shape)) {
              return {self_type->withDim(*list_size)};
            }
          }
          return {};
        }};
    static const register_formula_for reshape_broadcast_ops{
        {
            "aten::as_strided(Tensor self, int[] size, int[] stride, int? storage_offset) -> Tensor",
            "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
            "aten::repeat(Tensor self, int[] repeats) -> Tensor",
            "aten::view(Tensor self, int[] size) -> Tensor",
#if PYTORCH_VERSION_GE(1, 14)
            "prims::broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)",
#endif
        },
        [&](Node* node) -> type_vec_t {
          if (auto list_size = determineListSize(node->input(1))) {
            auto inpTy = node->input(0)->type()->cast<TensorType>();
            return {inpTy->withDim(*list_size)};
          }
          return {};
        }};

    // First, try to match one of the registered formulas to their operator
    // sets.
    for (auto& entry : shape_formulas) {
      if (entry.first.hasMember(*node)) {
        auto types = entry.second(node);
        if (types.empty()) {
          return false;
        } else {
          auto outputs = node->outputs();
          AT_ASSERT(types.size() == outputs.size());
          for (const auto i : c10::irange(types.size())) {
            AT_ASSERT(outputs[i]->type()->isSubtypeOf(TensorType::get()));
            outputs[i]->setType(types[i]);
          }
          return true;
        }
      }
    }

    // This section implements shape prop for an assorted set of nodes that only
    // need partial information about their input types.
    const auto input_type = [node](size_t index) {
      auto result = node->input(index)->type()->cast<TensorType>();
      if (result) {
        result = result->dimensionedOnly();
      }
      return result;
    };

    if (node->matches(
            "aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)") ||
        node->matches(
            "aten::topk(Tensor self, int k, int dim, bool largest, bool sorted) -> (Tensor, Tensor)")) {
      if (auto type = input_type(0)) {
        node->output(0)->setType(type);
        node->output(1)->setType(type->withScalarType(at::kLong));
        return true;
      }
#if PYTORCH_VERSION_GE(2, 1)
    } else if (
        node->matches(
            "aten::_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor ouput, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, int max_q, int max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)")) {
      if (auto q_type = input_type(0)) {
        node->output(0)->setType(q_type->withDim(4));
        node->output(1)->setType(
            q_type->withDim(3)->withScalarType(at::kFloat));
        node->output(2)->setType(q_type->withDim(1)->withScalarType(at::kInt));
        node->output(3)->setType(q_type->withDim(1)->withScalarType(at::kInt));
        node->output(6)->setType(
            q_type->withDim(0)->withScalarType(at::ScalarType::Long));
        node->output(7)->setType(
            q_type->withDim(0)->withScalarType(at::ScalarType::Long));
        node->output(8)->setType(q_type->withDim(0));
        return true;
      }
    } else if (
        node->matches(
            "aten::_scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, bool compute_log_sumexp, bool is_causal=False, *, float? scale=None) -> (Tensor output, Tensor log_sumexp)")) {
      if (auto q_type = input_type(0)) {
        node->output(0)->setType(q_type->withDim(4));
        node->output(1)->setType(
            q_type->withDim(3)->withScalarType(at::kFloat));
        return true;
      }
    } else if (
        node->matches(
            "aten::_scaled_dot_product_flash_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, int max_q, int max_k, float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, *, float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)") ||
        node->matches(
            "aten::_scaled_dot_product_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, bool is_causal=False, bool chunk_grad_outputs=False, *, float? scale=None) -> (Tensor, Tensor, Tensor)")) {
      if (auto q_type = input_type(1)) {
        if (auto k_type = input_type(2)) {
          node->output(0)->setType(q_type);
          node->output(1)->setType(k_type);
          node->output(2)->setType(k_type);
        }
        return true;
      }
#endif
    } else if (node->matches(
                   "aten::masked_select(Tensor self, Tensor mask) -> Tensor")) {
      if (auto type = input_type(0)) {
        node->output()->setType(type->withDim(1));
        return true;
      }
#if PYTORCH_VERSION_LE(1, 12)
    } else if (node->matches(
                   "aten::einsum(str equation, Tensor[] tensors) -> Tensor")) {
#else
    } else if (
        node->matches(
            "aten::einsum(str equation, Tensor[] tensors, *, int[]? path=None) -> Tensor")) {
#endif // PYTORCH_VERSION_LE(1, 12)
      auto equation = node->get<std::string>(attr::equation).value();
      auto found = equation.find("->");
      if (found == std::string::npos)
        return false;
      size_t rank = 0;
      while (found < equation.length()) {
        char ch = equation[found++];
        if ((ch <= 'z' && ch >= 'a') || (ch <= 'Z' && ch >= 'A'))
          rank++;
      }
      if (auto type = input_type(0)) {
        node->output()->setType(type->withDim(rank));
        return true;
      }
      return false;
    } else if (
        node->matches(
            "aten::tensor(t[] data, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)")) {
      auto outTy = node->output()->type()->cast<TensorType>();
      if (!(outTy && outTy->dim()))
        return false;
      auto list = node->input(0);
      if (auto list_size = determineListSize(list)) {
        std::vector<c10::ShapeSymbol> new_sizes =
            outTy->symbolic_sizes().sizes().value();
        new_sizes[0] = c10::ShapeSymbol::fromStaticSize(*list_size);
        node->output()->setType(outTy->withSymbolicShapes(new_sizes));
      }
      return true;
    } else if (node->matches(
                   "aten::__getitem__.t(t[](a) list, int idx) -> (t(*))")) {
      auto list_node = node->input(0)->node();
      if (list_node->kind() == aten::size)
        return true;

      if (list_node->kind() == aten::split) {
        if (auto typ = list_node->input(0)->type()->cast<TensorType>()) {
          if (typ->sizes().concrete_sizes().has_value()) {
            auto sizes = typ->sizes().concrete_sizes().value();
            auto idx = node->get<int>(attr::idx).value();
            auto dim = at::maybe_wrap_dim(
                list_node->get<int>(attr::dim).value(), sizes.size());
            auto split_size = list_node->get<int>(attr::split_size).value();
            int split_nums = sizes[dim] / split_size;
            if (sizes[dim] % split_size != 0)
              split_nums++;
            sizes[dim] = sizes[dim] - (split_nums - 1) * split_size;
            node->output()->setType(typ->withSizes(sizes));
          } else {
            auto original_type =
                list_node->input(0)->type()->cast<TensorType>();
            node->output()->setType(original_type->dimensionedOnly());
          }
          return true;
        }
      }
      if (list_node->kind() == aten::chunk) {
        auto original_type = list_node->input(0)->type()->cast<TensorType>();
        node->output()->setType(original_type->dimensionedOnly());
        return true;
      }
      if (list_node->kind() == aten::unbind) {
        auto original_type = list_node->input(0)->type()->cast<TensorType>();
        node->output()->setType(
            original_type->withDim(*original_type->dim() - 1));
        return true;
      }

      if (list_node->kind() != prim::ListConstruct)
        return false;
      auto tensors = list_node->inputs();
      auto idx = node->get<int>(attr::idx).value();
      if (tensors.size() <= idx)
        return false;
      node->output()->setType(tensors[idx]->type());
      return true;
    } else if (node->matches("aten::detach(Tensor(a) self) -> Tensor(a)")) {
      if (auto type = input_type(0)) {
        node->output()->setType(type->withRequiresGrad(false));
        return true;
      }
    } else if (
        node->matches(
            "aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)")) {
      if (auto type = input_type(0)) {
        if (type->scalarType() == at::kHalf) {
          type = type->withScalarType(at::kFloat);
        }
        type = type->withDim(1);
        node->outputs()[0]->setType(type);
        node->outputs()[1]->setType(type);
        return true;
      }
#if PYTORCH_VERSION_GE(1, 8)
    } else if (
        node->matches(
            "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")) {
      if (auto type = node->input(0)->type()->cast<TensorType>()) {
        auto output_mask =
            node->get<c10::List<bool>>(attr::output_mask).value();
        if (output_mask[0]) {
          node->outputs()[0]->setType(type);
        }
        if (output_mask[1]) {
          if (auto weight_type = node->input(5)->type()->cast<TensorType>()) {
            node->outputs()[1]->setType(weight_type);
          }
        }
        if (output_mask[2]) {
          if (auto bias_type = node->input(6)->type()->cast<TensorType>())
            node->outputs()[2]->setType(bias_type);
        }
        return true;
      }
    } else if (
        node->matches(
            "aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)")) {
      if (auto type = input_type(0)) {
        node->outputs()[0]->setType(type);
        if (auto m =
                determineListSize(node->namedInput(attr::normalized_shape))) {
          const size_t axis = type->dim().value() - m.value();
          std::vector<ShapeSymbol> stat_shape;
          for (const auto idx : c10::irange(axis)) {
            auto dimSize = getSymDimSize(type, idx);
            if (dimSize.is_static())
              // NB(xiafei.qiuxf): use static_size() rather than value() for
              // backward compatability. static_size() CHECKs is_static(), it's
              // safe here.
              stat_shape.emplace_back(
                  ShapeSymbol::fromStaticSize(dimSize.static_size()));
            else
              stat_shape.emplace_back(ShapeSymbol::newSymbol());
          }
          for (const auto idx C10_UNUSED :
               c10::irange(axis, type->dim().value())) {
            stat_shape.emplace_back(ShapeSymbol::fromStaticSize(1));
          }
          SymbolicShape symblicShape(stat_shape);
          node->outputs()[1]->setType(type->withSymbolicShapes(symblicShape));
          node->outputs()[2]->setType(type->withSymbolicShapes(symblicShape));
        }
      }
      return true;
#endif
    } else if (
        node->matches(
            "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)")) {
      if (auto type = input_type(0)) {
        node->outputs()[0]->setType(type);
        node->outputs()[1]->setType(type);
        node->outputs()[2]->setType(type);
        return true;
      }
    } else if (
        node->matches(
            "aten::index.Tensor_hacked_twin(Tensor self, Tensor?[] indices) -> Tensor")) {
      if (auto type = input_type(0)) {
        size_t max_rank = 0;
        if (auto indices = node->namedInput(attr::indices)) {
          for (auto input : indices->node()->inputs()) {
            if (input->type()->kind() == c10::TypeKind::NoneType)
              continue;
            if (auto input_type = input->type()->cast<TensorType>()) {
              max_rank = std::max(
                  max_rank,
                  input_type->symbolic_sizes().sizes().value().size());
            }
          }
          if (max_rank != 0) {
            node->outputs()[0]->setType(type->withDim(max_rank));
          }
          return true;
        }
      }
    } else if (
        node->matches(
            "aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)")) {
      int reduction = node->get<int>(attr::reduction).value();
      if (auto type = input_type(0)) {
        if (*type->dim() == 2 && reduction == 0) {
          node->outputs()[0]->setType(type->withDim(1));
        } else {
          node->outputs()[0]->setType(type->withDim(0));
        }
        node->outputs()[1]->setType(type->withDim(0));
        return true;
      }
#if PYTORCH_VERSION_GE(1, 14)
    } else if (
        node->matches(
            "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)")) {
      if (auto type = input_type(0)) {
        node->outputs()[0]->setType(type);
        node->outputs()[1]->setType(type->withScalarType(at::kBool));
        return true;
      }
    } else if (
        node->matches(
            "aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")) {
      if (auto mask = constant_as<c10::List<bool>>(
              node->namedInput(attr::output_mask))) {
        if (!std::all_of(
                mask->begin(), mask->end(), [](bool b) { return b; })) {
          return false;
        }
      }
      if (auto type = node->input(1)->type()->cast<TensorType>()) {
        if (type->sizes().concrete_sizes()) {
          type = type->withSizes(type->sizes().concrete_sizes().value());
        }
        node->output(0)->setType(type);
      }
      if (auto type = node->input(2)->type()->cast<TensorType>()) {
        node->output(1)->setType(type);
        if (auto shape = constant_as<c10::List<int64_t>>(
                node->namedInput(attr::bias_sizes))) {
          std::vector<int64_t> dims;
          for (size_t i = 0; i < shape.value().size(); ++i) {
            dims.push_back(shape.value()[i]);
          }
          node->output(2)->setType(type->withSizes(dims));
        }
      }
      return true;
    } else if (
        node->matches(
            "aten::upsample_nearest2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor")) {
      if (auto shape = constant_as<c10::List<int64_t>>(
              node->namedInput(attr::input_size))) {
        if (auto type = node->input(0)->type()->cast<TensorType>()) {
          std::vector<int64_t> dims;
          for (size_t i = 0; i < shape.value().size(); ++i) {
            dims.push_back(shape.value()[i]);
          }
          node->output(0)->setType(type->withSizes(dims));
        }
      }
      return true;
#endif
    } else if (node->matches(
                   "aten::dot(Tensor self, Tensor tensor) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(0));
        return true;
      }
    } else if (
        node->matches("aten::mv(Tensor self, Tensor vec) -> Tensor") ||
        node->matches(
            "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta, Scalar alpha) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(1));
        return true;
      }
    } else if (
        node->matches(
            "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta, Scalar alpha) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(2));
        return true;
      }
    } else if (
        node->matches(
            "aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta, Scalar alpha) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(3));
        return true;
      }
    } else if (
        node->matches(
            "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor")) {
      auto type = input_type(0);
      auto index_type = input_type(1);
      // index_select behaves very weirdly when self.dim() == 0. It allows both
      // 0D and 1D indices, and returns a value that has as many dimensions as
      // index.
      if (type && index_type && type->dim()) {
        if (*type->dim() == 0) {
          node->output()->setType(type->withDim(index_type->dim()));
        } else {
          node->output()->setType(type);
        }
        return true;
      }
    } else if (
        node->matches(
            "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor")) {
      auto type = input_type(0);
      auto index_type = input_type(2);
      // Gather has this annoying edge case where index always needs to match
      // the number of dims of self, **except** when self is 1D and index is 0D
      // in which case we return a 0D output.
      if (type && index_type && index_type->dim()) {
        if (*index_type->dim() == 0) {
          node->output()->setType(type->withDim(0));
        } else {
          node->output()->setType(type);
        }
        return true;
      }
    } else if (
        node->matches(
            "aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor")) {
      auto type = node->input(0)->type()->cast<TensorType>();
      if (type && type->dim()) {
        auto numWeightsOptional = node->get<int64_t>(attr::num_weights);
        if (type->isComplete()) {
          std::vector<int64_t> newSizes = {numWeightsOptional.value()};
          newSizes.push_back(type->sizes().concrete_sizes().value().back());
          node->output()->setType(type->withSizes(newSizes));
        } else {
          std::vector<ShapeSymbol> newSizes{
              ShapeSymbol::fromStaticSize(numWeightsOptional.value()),
              type->symbolic_sizes().sizes().value().back()};
          node->output()->setType(type->withSymbolicShapes(newSizes));
        }
        return true;
      }
    } else if (
        node->matches(
            "aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor")) {
      if (auto type = input_type(0)) {
        node->output()->setType(type);
        return true;
      }
      if (auto type = input_type(1)) {
        node->output()->setType(type);
        return true;
      }
    } else if (
        node->matches(
            "aten::dist(Tensor self, Tensor other, Scalar p) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(0));
        return true;
      }
    } else if (node->matches("aten::glu(Tensor self, int dim) -> Tensor")) {
      if (auto type = node->input(0)->type()->cast<TensorType>()) {
        auto sizesOptional = type->symbolic_sizes().sizes();
        auto dimOptional = node->get<int64_t>(attr::dim);
        if (!(sizesOptional && dimOptional))
          return false;

        std::vector<c10::ShapeSymbol> new_sizes = sizesOptional.value();
        int64_t input_rank = new_sizes.size();
        int64_t dim =
            at::maybe_wrap_dim(dimOptional.value(), input_rank, false);

        if (new_sizes[dim].is_static()) {
          new_sizes[dim] =
              ShapeSymbol::fromStaticSize(new_sizes[dim].static_size() / 2);
        } else {
          // set default to dynamic
          new_sizes[dim] = ShapeSymbol::newSymbol();
        }
        node->outputs()[0]->setType(type->withSymbolicShapes(new_sizes));
      }
      return true;
    } else if (
        node->matches(
#if PYTORCH_VERSION_GE(1, 14)
            "aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)"
#elif PYTORCH_VERSION_GE(1, 8)
            "aten::slice(Tensor self, int dim, int? start=None, int? end=None, int step=1) -> Tensor"
#else
            "aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor"
#endif
            )) {
      if (auto type = node->input(0)->type()->cast<TensorType>()) {
        auto sizesOptional = type->symbolic_sizes().sizes();
        auto dimOptional = node->get<int64_t>(attr::dim);
        if (!(sizesOptional && dimOptional))
          return false;

        std::vector<c10::ShapeSymbol> new_sizes = sizesOptional.value();
        int64_t input_rank = new_sizes.size();
        int64_t dim =
            at::maybe_wrap_dim(dimOptional.value(), input_rank, false);

        auto startOptional = node->get<IValue>(attr::start);
        auto endOptional = node->get<IValue>(attr::end);
        if (new_sizes[dim].is_static() && startOptional && endOptional) {
          int64_t start = startOptional.value() != c10::nullopt
              ? node->get<int64_t>(attr::start).value()
              : 0;
          int64_t end = endOptional.value() != c10::nullopt
              ? node->get<int64_t>(attr::end).value()
              : INT64_MAX;
          start =
              at::maybe_wrap_dim(start, new_sizes[dim].static_size(), false);
          int64_t step = node->get<int64_t>(attr::step).value();
          if (end >= new_sizes[dim].static_size())
            end = new_sizes[dim].static_size();
          int64_t len = end - start;
          new_sizes[dim] = ShapeSymbol::fromStaticSize((len + step - 1) / step);
        } else {
          // set default to dynamic
          new_sizes[dim] = ShapeSymbol::newSymbol();
        }
        node->outputs()[0]->setType(type->withSymbolicShapes(new_sizes));
      }
      return true;
    } else if (
        node->matches(
            "aten::embedding(Tensor weight, Tensor indices, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor")) {
      auto weight_type = input_type(0);
      auto indices_type = input_type(1);
      if (weight_type && indices_type && indices_type->dim()) {
        std::vector<ShapeSymbol> new_sizes =
            indices_type->symbolic_sizes().sizes().value();
        new_sizes.push_back(weight_type->symbolic_sizes().sizes().value()[1]);
        node->output()->setType(weight_type->withSymbolicShapes(new_sizes));
      }
      return true;
    }

    // The code below implements formulas that need type information for all
    // their tensor inputs, and have exactly one output.
    std::vector<TensorTypePtr> tensor_types;
    static const auto reshape_prop =
        [](Node* node,
           Symbol shape_input,
           const std::vector<TensorTypePtr>& tensor_types) -> TensorTypePtr {
      if (auto list_size = determineListSize(node->namedInput(shape_input))) {
        return tensor_types.at(0)->withDim(*list_size);
      }
      return nullptr;
    };

    const auto getSingleOutputType = [&]() -> TypePtr {
      if (node->matches("aten::type_as(Tensor self, Tensor other) -> Tensor")) {
        return tensor_types.at(0)->withScalarType(
            tensor_types.at(1)->scalarType());
      } else if (
          node->matches(
              "aten::view_as(Tensor(a) self, Tensor other) -> Tensor(a)") ||
          node->matches(
              "aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)") ||
          node->matches(
              "aten::reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)")) {
        return tensor_types.at(0)->withDim(tensor_types.at(1)->dim());
      } else if (
          node->matches(
              "aten::as_tensor(Tensor data, *, ScalarType? dtype, Device? device) -> Tensor")) {
        TypePtr input_type = node->inputs().at(0)->type();
        if (auto type = input_type->cast<TensorType>()) {
          if (type->scalarType() && type->device()) {
            at::ScalarType default_type = *type->scalarType();
            if (auto dtype_index =
                    node->schema().argumentIndexWithName("dtype")) {
              auto inp = toIValue(node->inputs().at(*dtype_index));
              if (inp == c10::nullopt) {
                return nullptr;
              }
              if (!inp->isNone()) {
                default_type = inp->toScalarType();
              }
            }
            auto device = getDeviceFromValue(node->namedInput(attr::device));
            node->output()->setType(TensorType::create(
                default_type,
                device,
                type->dim(),
                /*requires_grad=*/c10::nullopt));
          }
        }
        return nullptr;
      } else if (node->matches(
                     "aten::unsqueeze(Tensor self, int dim) -> Tensor")) {
        auto& t = tensor_types.at(0);
        if (!t->dim()) {
          return t;
        }
        return t->withDim(*t->dim() + 1);
      } else if (
          node->matches(
              "aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)",
              /*const_inputs=*/{attr::start_dim, attr::end_dim})) {
        auto& t = tensor_types.at(0);
        auto input_rank = *t->dim();
        if (input_rank == 0) {
          return t->withDim(1);
        }

        auto start_dim = at::maybe_wrap_dim(
            node->get<int>(attr::start_dim).value(), input_rank, false);
        auto end_dim = at::maybe_wrap_dim(
            node->get<int>(attr::end_dim).value(), input_rank, false);
        return t->withDim(input_rank - end_dim + start_dim);
      } else if (
          node->matches(
              "aten::select(Tensor self, int dim, int index) -> Tensor") ||
          node->matches(
              "aten::diagonal(Tensor self, int offset, int dim1, int dim2) -> Tensor")) {
        auto& t = tensor_types.at(0);
        return t->dim() && *t->dim() > 0 ? t->withDim(*t->dim() - 1) : nullptr;
      } else if (
          node->matches(
              "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor")) {
        if (!tensor_types.at(0)->dim() || !tensor_types.at(1)->dim()) {
          return nullptr;
        }
        auto inpTy0 = tensor_types.at(0);
        auto inpTy1 = tensor_types.at(1);
        int dim1 = *inpTy0->dim();
        int dim2 = *inpTy1->dim();
        if (dim2 != 2 || dim1 == 0) {
          return nullptr;
        }

        std::vector<ShapeSymbol> stat_shape;
        for (auto d = 0; d < dim1; ++d) {
          stat_shape.push_back(getSymDimSize(inpTy0, d));
        }
        stat_shape[dim1 - 1] = getSymDimSize(inpTy1, 0);
        SymbolicShape symblicShape(stat_shape);
        return inpTy0->withSymbolicShapes(symblicShape);
      } else if (node->matches(
                     "aten::matmul(Tensor self, Tensor other) -> Tensor")) {
        if (!tensor_types.at(0)->dim() || !tensor_types.at(1)->dim()) {
          return nullptr;
        }

        std::vector<ShapeSymbol> stat_shape;
        auto inpTy0 = tensor_types.at(0);
        auto inpTy1 = tensor_types.at(1);
        int dim1 = *inpTy0->dim();
        int dim2 = *inpTy1->dim();
        if (dim1 == 1 && dim2 == 1) {
          // Dot product
          return inpTy0->withDim(0);
          // NOLINTNEXTLINE(bugprone-branch-clone)
        } else if (dim1 == 2 && dim2 == 2) {
          // Matrix multiply
          stat_shape.push_back(getSymDimSize(inpTy0, 0));
          stat_shape.push_back(getSymDimSize(inpTy1, 1));
          SymbolicShape symblicShape(stat_shape);
          return inpTy0->withSymbolicShapes(symblicShape);
        } else if (dim1 == 1 && dim2 == 2) {
          // Unsqueeze + matrix multiply + squeeze
          stat_shape.push_back(getSymDimSize(inpTy1, 1));
          SymbolicShape symblicShape(stat_shape);
          return inpTy0->withSymbolicShapes(symblicShape);
        } else if (dim1 == 2 && dim2 == 1) {
          // Matrix vector multiply
          stat_shape.push_back(getSymDimSize(inpTy0, 0));
          SymbolicShape symblicShape(stat_shape);
          return inpTy0->withSymbolicShapes(symblicShape);
        } else {
          // Batched matrix multiply (possibly with squeeze + unsqueeze if one
          // argument is 1D)
          auto type = broadcast(tensor_types, tensor_types[0]->scalarType());
          if (dim1 == 1 || dim2 == 1) {
            type = type->withDim(type->dim().value() - 1);
          }
          return type;
        }
      } else if (node->matches("aten::nonzero(Tensor self) -> Tensor")) {
        return tensor_types.at(0)->dimensionedOnly()->withScalarType(at::kLong);
      } else if (node->matches(
                     "aten::take(Tensor self, Tensor index) -> Tensor")) {
        return tensor_types.at(1)->dimensionedOnly()->withScalarType(
            tensor_types.at(0)->scalarType());
      } else if (node->matches(
                     "aten::diagflat(Tensor self, int offset) -> Tensor")) {
        return tensor_types.at(0)->withDim(2);
      } else if (node->matches(
                     "aten::diag(Tensor self, int diagonal) -> Tensor")) {
        auto& t = tensor_types.at(0);
        if (t->dim() && *t->dim() == 1) {
          return t->withDim(2);
        } else if (t->dim() && *t->dim() == 2) {
          return t->withDim(1);
        } else {
          return nullptr;
        }
      } else if (
          node->matches(
              "aten::unfold(Tensor self, int dimension, int size, int step) -> Tensor")) {
        auto& t = tensor_types.at(0);
        if (!t->dim()) {
          return nullptr;
        }
        return t->withDim(*t->dim() + 1);
      } else if (node->matches(
                     "aten::polygamma(int n, Tensor self) -> Tensor")) {
        return tensor_types.at(0);
      }
      return nullptr;
    };
    if (auto maybe_tensor_types = gatherTensorTypes(node)) {
      tensor_types = std::move(*maybe_tensor_types);
    } else {
      return false;
    }
    if (node->outputs().size() == 1) {
      if (auto type = getSingleOutputType()) {
        node->output()->setType(type);
        return true;
      }
    }
    return false;
  }
#if PYTORCH_VERSION_GE(1, 14)
  bool PropagateCompleteShapeOnNode(
      Node* node,
      bool insert_expands,
      std::vector<TensorTypePtr> tensor_types) {
    // For expensive ops we can directly encode their shape propagation
    // here, otherwise we fallback to running a fake version of the op
    // to get a quick and dirty propagation.
    if (node->matches(
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        node->matches("aten::mul(Tensor self, Tensor other) -> Tensor")) {
      // These nodes handle tensors of different shapes internally, so there's
      // no need to insert explicit expand nodes.
      return PropagateShapeOnNodeByRunningIt(node);
    } else if (node->matches(
                   "aten::div(Tensor self, Tensor other) -> Tensor")) {
      // "div" handle tensors of different shapes internally, so there's no need
      // to insert explicit expand nodes.
      // Note that this function could be merged to the one above , but "div" is
      // not always safe to run by itself due to integer divide-by-zero.
      // We fake the execution by running "mul" operation instead.
      auto op = getOperatorForLiteral(
                    "aten::mul(Tensor self, Tensor other) -> Tensor")
                    ->getOperation();
      return PropagateShapeOnNodeByRunningIt(node, std::move(op));
    } else if (
        node->matches(
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
        node->matches("aten::div(Tensor self, Scalar other) -> Tensor") ||
        node->matches("aten::mul(Tensor self, Scalar other) -> Tensor")) {
      auto first_scalar_type = (tensor_types)[0]->scalarType();
      auto second_scalar_type =
          tryScalarTypeFromJitType(*node->inputs()[1]->type());
      if (!first_scalar_type || !second_scalar_type) {
        return false;
      }
      if (isIntegralType(*first_scalar_type, false) &&
          isFloatingType(*second_scalar_type)) {
        auto default_dtype =
            at::typeMetaToScalarType(caffe2::get_default_dtype());
        auto type = tensor_types[0]->withScalarType(default_dtype);
        node->output()->setType(std::move(type));
        return true;
      }
      if (c10::ScalarType::Bool == *first_scalar_type &&
          c10::ScalarType::Bool != *second_scalar_type) {
        auto result_type =
            c10::promoteTypes(*first_scalar_type, *second_scalar_type);
        auto type = tensor_types[0]->withScalarType(result_type);
        node->output()->setType(std::move(type));
        return true;
      }
      auto type = tensor_types[0]->withScalarType(first_scalar_type);
      node->output()->setType(std::move(type));
      return true;
    } else if (
        insert_expands &&
        (node->matches("aten::min(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::max(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::lt(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::le(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::gt(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::ge(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::eq(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::ne(Tensor self, Tensor other) -> Tensor"))) {
      // Binary broadcasting ops
      // NB: we don't handle the nodes in any other way (note the lack of
      // return!), because the tpe casting logic in scalar cases is
      // non-trivial. It's better to just run them.
      broadcastBinary(node, tensor_types, 0, 1);
      return PropagateShapeOnNodeByRunningIt(node);
    } else if (
        node->matches(
            "aten::logit(Tensor self, float? eps = None) -> Tensor") ||
        node->matches("aten::neg(Tensor self) -> Tensor") ||
        node->matches("aten::sigmoid(Tensor self) -> Tensor") ||
        node->matches("aten::tanh(Tensor self) -> Tensor")) {
      node->output()->setType(tensor_types.at(0)->contiguous());
      return true;
    } else if (node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
      auto lhs_type = tensor_types.at(0);
      auto rhs_type = tensor_types.at(1);
      auto lhs_sizes = lhs_type->sizes().concrete_sizes().value();
      auto rhs_sizes = rhs_type->sizes().concrete_sizes().value();
      SHAPE_ASSERT(
          *lhs_type->sizes().size() == 2 && *rhs_type->sizes().size() == 2);
      node->output()->setType(TensorType::createContiguous(
          *lhs_type->scalarType(),
          *lhs_type->device(),
          at::IntArrayRef{lhs_sizes[0], rhs_sizes[1]}));
      return true;
    } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
      auto tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      auto strides = tp->strides().concrete_sizes().value();
      SHAPE_ASSERT(sizes.size() == 2);
      std::swap(sizes.at(0), sizes.at(1));
      std::swap(strides.at(0), strides.at(1));
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      return true;
    } else if (
        node->matches(
            "aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
            /*const_inputs=*/{attr::dim, attr::length})) {
      auto tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      int64_t dim = node->get<int64_t>(attr::dim).value();
      int64_t length = node->get<int64_t>(attr::length).value();
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
      sizes.at(dim) = length;
      node->output()->setType(
          tp->withSizesStrides(sizes, tp->strides().concrete_sizes().value()));
      return true;
    } else if (node->matches(
                   "aten::sum(Tensor self, *, int? dtype) -> Tensor")) {
      node->output()->setType(tensor_types.at(0)->withSizes({}));
      return true;
    } else if (
        node->matches(
            "aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
            /*const_inputs=*/{attr::dim, attr::keepdim})) {
      auto& tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      auto dims = node->get<c10::List<int64_t>>(attr::dim).value();
      bool keepdim = node->get<bool>(attr::keepdim).value();
      std::reverse(dims.begin(), dims.end());
      std::vector<int64_t> new_dims;
      for (size_t i = 0; i < sizes.size(); ++i) {
        if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
          new_dims.push_back(sizes.at(i));
        } else if (keepdim) {
          new_dims.push_back(1);
        }
      }
      node->output()->setType(tp->withSizes(new_dims));
      return true;
    } else if (node->matches(
                   "aten::unsqueeze(Tensor self, int dim) -> Tensor",
                   /*const_inputs=*/attr::dim)) {
      auto& tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) <= sizes.size());
      sizes.insert(sizes.begin() + dim + 1, 1);
      node->output()->setType(tp->withSizes(sizes));
      return true;
    } else if (node->matches(
                   "aten::view(Tensor self, int[] size) -> Tensor",
                   /*const_inputs=*/attr::size)) {
      auto sizes = node->get<c10::List<int64_t>>(attr::size).value();
      bool inferred = false;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      size_t inferred_idx;
      int64_t size_product = 1;
      for (const auto i : c10::irange(sizes.size())) {
        if (sizes.get(i) == -1) {
          if (inferred)
            throw propagation_error();
          inferred = true;
          inferred_idx = i;
        } else {
          size_product *= sizes.get(i);
        }
      }

      if (inferred) {
        SHAPE_ASSERT(size_product != 0);
        size_t numel = 1;
        auto concrete_sizes =
            tensor_types.at(0)->sizes().concrete_sizes().value();
        for (int64_t s : concrete_sizes)
          numel *= s;
        int64_t inferred_size = numel / size_product;
        sizes[inferred_idx] = inferred_size;
      }
      node->output()->setType(tensor_types.at(0)->withSizes(sizes.vec()));
      return true;
    } else if (node->matches(
                   "aten::type_as(Tensor self, Tensor other) -> Tensor")) {
      if (tensor_types.at(0)->scalarType() ==
          tensor_types.at(1)->scalarType()) {
        node->output()->setType(node->namedInput(attr::self)->type());
      } else {
        // This will be a copy, so the result will be contiguous
        node->output()->setType(tensor_types.at(1)->withSizes(
            tensor_types.at(0)->sizes().concrete_sizes().value()));
      }
      return true;
#if PYTORCH_VERSION_GE(1, 13)
    } else if (
        node->matches(
            "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
            /*const_inputs=*/attr::size)) {
      auto tp = tensor_types.at(0);
      auto sizesAndStrides = at::inferExpandGeometry_dimvector(
          tp->sizes().concrete_sizes().value(),
          tp->strides().concrete_sizes().value(),
          node->get<c10::List<int64_t>>(attr::size).value().vec());
      node->output()->setType(
          tp->withSizesStrides(sizesAndStrides.sizes, sizesAndStrides.strides));
      return true;
#endif
    } else if (
        node->matches(
            "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor",
            /*const_inputs=*/attr::dim)) {
      auto ten = tensor_types.at(0);
      auto index = tensor_types.at(1);
      int64_t dim = node->get<int64_t>(attr::dim).value();
      SHAPE_ASSERT(*index->sizes().size() == 1);
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < ten->sizes().size());
      std::vector<int64_t> sizes = ten->sizes().concrete_sizes().value();
      sizes[dim] = index->sizes()[0].value();
      node->output()->setType(ten->withSizes(sizes));
      return true;
    } else if (node->matches(
                   "aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]",
                   /*const_inputs=*/{attr::chunks, attr::dim})) {
      auto input_type = tensor_types.at(0);
      auto sizes = input_type->sizes().concrete_sizes().value();
      auto strides = input_type->strides().concrete_sizes().value();
      int64_t dim = node->get<int64_t>(attr::dim).value();
      int64_t chunks = node->get<int64_t>(attr::chunks).value();
      sizes[dim] /= chunks;
      for (Value* output : node->outputs()) {
        output->setType(input_type->withSizesStrides(sizes, strides));
      }
      if (*input_type->sizes()[dim] % chunks != 0) {
        sizes[dim] = *input_type->sizes()[dim] % chunks;
        node->outputs().back()->setType(
            input_type->withSizesStrides(sizes, strides));
      }
      return true;
    }
    setUnshapedType(node);
    return false;
  }
#endif
};
} // anonymous namespace

void PropagateInputShapes(const std::shared_ptr<Graph>& graph) {
  // Before shape propagation, one must guarantee that graph
  // is in SSA mode. Thus there are no mutations in the graph.
  ShapePropagator(graph).propagateBlock(graph->block());
}

namespace {

using TypeCache = std::unordered_map<TypePtr, TypePtr>;

TypePtr getOrCreateUnshapedType(TypePtr type, TypeCache& unshaped_type_cache);

TypePtr unshapedTypeImpl(TypePtr type, TypeCache& unshaped_type_cache) {
  if (type->isSubtypeOf(TensorType::get())) {
    return TensorType::get();
  }
  at::ArrayRef<TypePtr> contained = type->containedTypes();
  if (contained.empty()) {
    return type;
  }
  std::vector<TypePtr> unshaped_contained_types;
  for (const auto& contained_type : contained) {
    unshaped_contained_types.push_back(
        getOrCreateUnshapedType(contained_type, unshaped_type_cache));
  }
  return type->withContained(unshaped_contained_types);
}

TypePtr getOrCreateUnshapedType(TypePtr type, TypeCache& unshaped_type_cache) {
  auto maybe_cached_type = unshaped_type_cache.find(type);
  if (maybe_cached_type != unshaped_type_cache.end()) {
    return maybe_cached_type->second;
  }
  auto unshaped_type = unshapedTypeImpl(type, unshaped_type_cache);
  unshaped_type_cache[type] = unshaped_type;
  return unshaped_type;
}

void EraseShapeInformation(
    const std::shared_ptr<Graph>& graph,
    TypeCache& unshaped_type_cache);

void EraseShapeInformation(
    at::ArrayRef<Value*> vals,
    TypeCache& unshaped_type_cache) {
  for (Value* v : vals) {
    v->setType(getOrCreateUnshapedType(v->type(), unshaped_type_cache));
  }
}

void EraseShapeInformation(Block* b, TypeCache& unshaped_type_cache) {
  EraseShapeInformation(b->inputs(), unshaped_type_cache);
  EraseShapeInformation(b->outputs(), unshaped_type_cache);
  for (Node* n : b->nodes()) {
    EraseShapeInformation(n->outputs(), unshaped_type_cache);
    for (Block* sb : n->blocks()) {
      EraseShapeInformation(sb, unshaped_type_cache);
    }
    if (n->hasAttribute(attr::Subgraph)) {
      EraseShapeInformation(n->g(attr::Subgraph), unshaped_type_cache);
    }
  }
}

void EraseShapeInformation(
    const std::shared_ptr<Graph>& graph,
    TypeCache& unshaped_type_cache) {
  EraseShapeInformation(graph->block(), unshaped_type_cache);
}

} // anonymous namespace

void EraseShapeInformation(const std::shared_ptr<Graph>& graph) {
  TypeCache unshaped_type_cache;
  EraseShapeInformation(graph->block(), unshaped_type_cache);
}
} // namespace blade
} // namespace torch
