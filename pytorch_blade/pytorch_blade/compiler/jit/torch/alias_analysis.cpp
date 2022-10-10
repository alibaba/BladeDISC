// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pytorch_blade/compiler/jit/torch/alias_analysis.h"

namespace torch {
namespace blade {
using namespace ::torch::jit;
namespace {
// For any mutable type, map it to a type such that all other types which it can
// alias will be mapped to the same type. This function follows a similar logic
// to `unifyTypes` because any two mutable types which can be unified
// can alias each other.
// getMutableTypePtr(Optional[List[int]]) == getMutableTypePtr([List[int]])
// If a type is not mutable, return nullopt
c10::optional<TypePtr> getMutableType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ListType:
    case TypeKind::DictType:
    case TypeKind::ClassType:
    case TypeKind::TensorType:
      // TODO: lookup cached contained types. this is kind of tricky
      // because a List[Optional[T]] should still be
      // List[Optional[Unshaped(T)]], however the getMutableType(Optional[T])
      // == T
      return unshapedType(type);
    case TypeKind::OptionalType:
      return getMutableType(type->cast<OptionalType>()->getElementType());
    case TypeKind::AnyType:
      return type;
    case TypeKind::FutureType: {
      if (auto elem =
              getMutableType(type->cast<FutureType>()->getElementType())) {
        return FutureType::create(*elem);
      }
      return c10::nullopt;
    }
    case TypeKind::TupleType: {
      std::vector<TypePtr> mutable_types;
      for (const auto& elem : type->expect<TupleType>()->elements()) {
        if (auto mut_elem = getMutableType(elem)) {
          mutable_types.push_back(*mut_elem);
        }
      }
      if (mutable_types.size() == 0) {
        return c10::nullopt;
      } else {
        return TupleType::create(mutable_types);
      }
    }
    default:
      return c10::nullopt;
  }
}

bool isMutableTypeImpl(const TypePtr& type) {
  // check common cases to avoid recursively constructing type in
  // getMutableTypePtrImpl
  auto kind = type->kind();
  if (kind == TypeKind::TensorType || kind == TypeKind::ListType ||
      kind == TypeKind::ClassType || kind == TypeKind::DictType) {
    return true;
  }
  return getMutableType(type) != c10::nullopt;
}
} // namespace

bool isMutableType(const TypePtr& type) {
  return isMutableTypeImpl(type);
}

bool isMutableType(const Value* v) {
  return isMutableType(v->type());
}
} // namespace blade
} // namespace torch
