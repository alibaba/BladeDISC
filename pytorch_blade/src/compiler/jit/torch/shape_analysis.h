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

#pragma once

#include <c10/util/ArrayRef.h>
#include <memory>

namespace torch {
namespace jit {
struct Graph;
struct Block;
struct Node;
struct Value;
} // namespace jit
} // namespace torch

namespace torch {
namespace blade {

using torch::jit::Block;
using torch::jit::Graph;
using torch::jit::Node;
using torch::jit::Value;

struct propagation_error : std::exception {};

class PropertyPropBase {
  // Used for both Shape Propagation and Dtype/Device Propagation
 public:
  explicit PropertyPropBase(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}
  virtual ~PropertyPropBase() = default;

  void propagateBlock(torch::jit::Block* block, bool insert_expands = true);
  // insert_expands is used for shape inference

  void processIf(torch::jit::Node* node);
  void processLoop(torch::jit::Node* node);

 protected:
  virtual void propagateNode(
      torch::jit::Node* node,
      bool insert_expands = true) = 0;
  void setUnshapedType(Value* o);
  void setUnshapedType(torch::jit::Node* node);
  std::shared_ptr<torch::jit::Graph> graph_;
};

void EraseShapeInformation(const std::shared_ptr<torch::jit::Graph>& graph);
void PropagateInputShapes(const std::shared_ptr<torch::jit::Graph>& graph);

bool mergeTypes(
    c10::ArrayRef<Value*> lhs,
    c10::ArrayRef<Value*> rhs,
    c10::ArrayRef<Value*> outputs);

} // namespace blade
} // namespace torch
