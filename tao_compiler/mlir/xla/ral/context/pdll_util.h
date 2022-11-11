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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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

  // Returns the type of the attr
  std::string& getValue() { return str_; }

 private:
  std::string str_;
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

// Parse the serialized attr from buffer. Returns nullptr if failed.
std::unique_ptr<PDLAttr> parsePDLAttr(uint8_t*& buffer);

}  // namespace ral
}  // namespace tao

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_PDLL_UTIL_H_
