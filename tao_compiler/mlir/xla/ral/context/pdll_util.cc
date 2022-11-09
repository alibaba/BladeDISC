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

#include "tensorflow/compiler/mlir/xla/ral/context/pdll_util.h"

#include "tensorflow/compiler/mlir/xla/ral/ral_logging.h"

namespace tao {
namespace ral {

template <typename T>
T parsePOD(uint8_t*& data) {
  T r = *(T*)data;
  data += sizeof(T);
  return r;
}

std::string parseStr(uint8_t*& data) {
  auto bytes = parsePOD<int64_t>(data);
  auto basePtr = (const char*)(data);
  std::string r(basePtr, basePtr + bytes);
  data += bytes;
  return r;
}

std::unique_ptr<PDLAttr> parseDictAttr(uint8_t*& buffer) {
  std::string type = parseStr(buffer);
  assert(type == "dict");
  auto nElems = parsePOD<int64_t>(buffer);
  TAO_VLOG(1) << "parseDictAttr try to parse with " << nElems << " elements";
  auto dictAttr = std::make_unique<DictPDLAttr>(type);
  for (int64_t i = 0; i < nElems; ++i) {
    auto key = parseStr(buffer);
    auto attrPtr = parsePDLAttr(buffer);
    if (!attrPtr) return attrPtr;
    TAO_VLOG(1) << "\t#" << i << ": " << key << "@" << attrPtr->getType();
    dictAttr->insert(key, std::move(attrPtr));
  }
  return dictAttr;
}

std::unique_ptr<PDLAttr> parseStrAttr(uint8_t*& buffer) {
  std::string type = parseStr(buffer);
  assert(type == "str");
  std::string data = parseStr(buffer);
  TAO_VLOG(1) << "parseStrAttr parsed str = " << data;
  return std::make_unique<StrPDLAttr>(type, data);
}

std::unique_ptr<PDLAttr> parsePDLAttr(uint8_t*& buffer) {
  uint8_t* tryBuffer = buffer;
  std::string type = parseStr(tryBuffer);
  TAO_VLOG(1) << "parsePDLAttr try to parse attr with type " << type;
  if (type == "dict") {
    return parseDictAttr(buffer);
  } else if (type == "str") {
    return parseStrAttr(buffer);
  }
  return nullptr;
}

}  // namespace ral
}  // namespace tao
