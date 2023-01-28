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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_PDLL_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_PDLL_UTIL_H_

#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlir/xla/ral/ral_context.h"

namespace tao {
namespace ral {

class PDLAttr {
 public:
  explicit PDLAttr(const std::string& type) : type_(type) {}

  virtual ~PDLAttr() {}

  // Returns the type of the attr
  const std::string& getType() const { return type_; }

  template <typename T>
  T& as() {
    return *(T*)(this);
  }

 private:
  std::string type_;
};

using PDLAttrPtr = std::unique_ptr<PDLAttr>;

class StrPDLAttr : public PDLAttr {
 public:
  explicit StrPDLAttr(const std::string& type, const std::string& str)
      : PDLAttr(type), str_(str) {}

  // Returns the value of the attr
  std::string& getValue() { return str_; }

 private:
  std::string str_;
};

template <typename T>
class PodPDLAttr : public PDLAttr {
 public:
  explicit PodPDLAttr(const std::string& type, T val)
      : PDLAttr(type), val_(val) {}

  // Returns the value of the attr
  T getValue() { return val_; }

 private:
  T val_;
};

using BoolPDLAttr = PodPDLAttr<bool>;
using IntPDLAttr = PodPDLAttr<int64_t>;
using FloatPDLAttr = PodPDLAttr<double>;

class DenseElementsPDLAttr : public PDLAttr {
 public:
  explicit DenseElementsPDLAttr(const std::string& type,
                                const std::string& elemTy, unsigned numBits,
                                std::vector<int64_t> shape,
                                std::vector<uint8_t> rawData)
      : PDLAttr(type),
        elemTy_(elemTy),
        numBits_(numBits),
        shape_(std::move(shape)),
        rawData_(std::move(rawData)) {}

  // Returns raw data of the attr
  std::vector<uint8_t>& getRawData() { return rawData_; }

  template <typename T>
  T* getValue() {
    return (T*)rawData_.data();
  }

  // Returns the shape of the denseElementsAttr
  std::vector<int64_t> getShape() { return shape_; }

  // Returns the element type of the denseElementsAttr.
  const std::string& getElementType() { return elemTy_; }

  // Returns the num of bits of the element type of the dense attr.
  unsigned getNumBits() { return numBits_; }

  // Returns the total number of elements of the denseElementsAttr.
  int64_t getNumElements() {
    int numElems = 1;
    for (int64_t d : shape_) numElems *= d;
    return numElems;
  }

 private:
  std::string elemTy_;
  unsigned numBits_;
  std::vector<int64_t> shape_;
  std::vector<uint8_t> rawData_;
};

class DictPDLAttr : public PDLAttr {
 public:
  using PDLAttr::PDLAttr;

  // Returns true if such dict has an item named `key`.
  bool hasKey(const std::string& key) const {
    return namedAttrs_.find(key) != namedAttrs_.end();
  };

  // Returns the value indexed by `key`. The key must be in the dict.
  PDLAttr& get(const std::string& key) {
    auto it = namedAttrs_.find(key);
    assert(it != namedAttrs_.end());
    return *it->second;
  };

  // Inserts a new pair <key, value>.
  // Returns false if failed (e.g. duplicated key).
  bool insert(const std::string& key, PDLAttrPtr value) {
    return namedAttrs_.emplace(key, std::move(value)).second;
  }

  std::unordered_map<std::string, PDLAttrPtr>& getValue() {
    return namedAttrs_;
  }

 private:
  std::unordered_map<std::string, PDLAttrPtr> namedAttrs_;
};

class ArrayPDLAttr : public PDLAttr {
 public:
  using PDLAttr::PDLAttr;

  PDLAttr& get(int64_t index) {
    assert(index >= 0 && index < arrayAttr_.size());
    return *(arrayAttr_[index]);
  }

  uint64_t size() { return arrayAttr_.size(); }

  void push_back(PDLAttrPtr value) { arrayAttr_.push_back(std::move(value)); }

  std::vector<PDLAttrPtr>& getValue() { return arrayAttr_; }

 private:
  std::vector<PDLAttrPtr> arrayAttr_;
};

class IntArrayPDLAttr : public PDLAttr {
 public:
  using PDLAttr::PDLAttr;

  int64_t get(int64_t index) {
    assert(index >= 0 && index < intArrayAttr_.size());
    return intArrayAttr_[index];
  }

  uint64_t size() { return intArrayAttr_.size(); }

  void push_back(int64_t val) { intArrayAttr_.push_back(val); }

  std::vector<int64_t>& getValue() { return intArrayAttr_; }

 private:
  std::vector<int64_t> intArrayAttr_;
};

// Parse the serialized attr from buffer. Returns nullptr if failed.
std::unique_ptr<PDLAttr> parsePDLAttr(uint8_t*& buffer);

template <typename T>
struct CustomAttrState : public Context::Resource {
  std::mutex mu;
  std::unordered_map<void*, std::unique_ptr<T>> customAttrMap;
};

template <typename T>
inline T* getOrParseCustomAttr(ExecutionContext* ctx, void* attrPtr,
                               const std::string& name,
                               std::function<std::unique_ptr<T>()> creator) {
  using StateTy = CustomAttrState<T>;
  auto state =
      ctx->getOrCreateResource<StateTy>(name, [&]() { return new StateTy; });
  std::lock_guard<std::mutex> l(state->mu);
  auto it = state->customAttrMap.find(attrPtr);
  if (it == state->customAttrMap.end()) {
    auto value = creator();
    if (!value) return nullptr;
    it = state->customAttrMap.emplace(attrPtr, std::move(value)).first;
  }
  return it->second.get();
}

inline PDLAttr* getOrParsePDLAttr(ExecutionContext* ctx, void* attrPtr,
                                  const std::string& name) {
  auto creator = [&]() {
    auto ptr = (uint8_t*)(attrPtr);
    return parsePDLAttr(ptr);
  };
  return getOrParseCustomAttr<PDLAttr>(ctx, attrPtr, name, creator);
}

}  // namespace ral
}  // namespace tao

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_PDLL_UTIL_H_
