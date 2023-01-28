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

#include "mlir/xla/ral/context/pdll_util.h"

#include "mlir/xla/ral/ral_logging.h"

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

std::unique_ptr<PDLAttr> parseBoolAttr(uint8_t*& buffer) {
  std::string type = parseStr(buffer);
  assert(type == "bool");
  bool val = parsePOD<bool>(buffer);
  TAO_VLOG(1) << "parseBoolAttr parsed val = " << val;
  return std::make_unique<BoolPDLAttr>(type, val);
}

std::unique_ptr<PDLAttr> parseIntAttr(uint8_t*& buffer) {
  std::string type = parseStr(buffer);
  assert(type == "int");
  int64_t val = parsePOD<int64_t>(buffer);
  TAO_VLOG(1) << "parseIntAttr parsed value = " << val;
  return std::make_unique<IntPDLAttr>(type, val);
}

std::unique_ptr<PDLAttr> parseFloatAttr(uint8_t*& buffer) {
  std::string type = parseStr(buffer);
  assert(type == "float");
  double val = parsePOD<double>(buffer);
  TAO_VLOG(1) << "parseFloatAttr parsed value = " << val;
  return std::make_unique<FloatPDLAttr>(type, val);
}

std::unique_ptr<PDLAttr> parseDenseElementsAttr(uint8_t*& buffer) {
  std::string type = parseStr(buffer);
  assert(type == "denseElementsAttr");

  // parse element type
  std::string elemTy = parseStr(buffer);
  // parse num bits of element type
  unsigned numBits = parsePOD<unsigned>(buffer);
  // parse the rank of the DenseElementsAttr
  int64_t rank = parsePOD<int64_t>(buffer);
  // parse the shape of the DenseElementsAttr
  std::vector<int64_t> shape;
  shape.reserve(rank);
  int64_t numElems = 1;
  for (int i = 0; i < rank; ++i) {
    shape.push_back(parsePOD<int64_t>(buffer));
    numElems *= shape.back();
  }
  bool isSplat = parsePOD<bool>(buffer);
  std::string data = parseStr(buffer);
  uint8_t* dataPtr = (uint8_t*)(data.data());
  std::vector<uint8_t> rawData(dataPtr, dataPtr + data.size());
  if (isSplat) {
    TAO_CHECK(rawData.size() == (numBits / 8));
    std::vector<uint8_t> newRawData(numElems * numBits / 8);
    for (size_t i = 0; i < newRawData.size(); ++i) {
      newRawData[i] = rawData[i % rawData.size()];
    }
    rawData = std::move(newRawData);
  }
  TAO_VLOG(1) << "parseDenseElementsAttr type@" << elemTy << numBits
              << ", shape = [";
  for (int64_t v : shape) TAO_VLOG(1) << "\t" << v;
  TAO_VLOG(1) << "]";
  return std::make_unique<DenseElementsPDLAttr>(
      type, elemTy, numBits, std::move(shape), std::move(rawData));
}

std::unique_ptr<PDLAttr> parseArrayAttr(uint8_t*& buffer) {
  std::string type = parseStr(buffer);
  assert(type == "array");
  auto nElems = parsePOD<int64_t>(buffer);
  TAO_VLOG(1) << "parseArrayAttr try to parse with " << nElems << " elements";
  auto arrayAttr = std::make_unique<ArrayPDLAttr>(type);
  for (uint64_t i = 0; i < nElems; ++i) {
    auto attrPtr = parsePDLAttr(buffer);
    if (!attrPtr) return attrPtr;
    arrayAttr->push_back(std::move(attrPtr));
  }
  return arrayAttr;
}

std::unique_ptr<PDLAttr> parseIntArrayAttr(uint8_t*& buffer) {
  std::string type = parseStr(buffer);
  assert(type == "intArray");
  auto nElems = parsePOD<int64_t>(buffer);
  TAO_VLOG(1) << "parseIntArrayAttr try to parse with " << nElems
              << " elements";
  auto intArrayAttr = std::make_unique<IntArrayPDLAttr>(type);
  for (uint64_t i = 0; i < nElems; ++i) {
    int64_t val = parsePOD<int64_t>(buffer);
    intArrayAttr->push_back(val);
  }
  return intArrayAttr;
}

std::unique_ptr<PDLAttr> parsePDLAttr(uint8_t*& buffer) {
  uint8_t* tryBuffer = buffer;
  std::string type = parseStr(tryBuffer);
  TAO_VLOG(1) << "parsePDLAttr try to parse attr with type " << type;
  if (type == "dict") {
    return parseDictAttr(buffer);
  } else if (type == "str") {
    return parseStrAttr(buffer);
  } else if (type == "bool") {
    return parseBoolAttr(buffer);
  } else if (type == "int") {
    return parseIntAttr(buffer);
  } else if (type == "float") {
    return parseFloatAttr(buffer);
  } else if (type == "denseElementsAttr") {
    return parseDenseElementsAttr(buffer);
  } else if (type == "array") {
    return parseArrayAttr(buffer);
  } else if (type == "intArray") {
    return parseIntArrayAttr(buffer);
  }
  return nullptr;
}

}  // namespace ral
}  // namespace tao
