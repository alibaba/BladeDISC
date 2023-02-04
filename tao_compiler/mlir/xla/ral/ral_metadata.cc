//===- ral_logging.h ----------------------===//
//
// Copyright 2022 The PAI Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/xla/ral/ral_metadata.h"

#include <unordered_map>

namespace tao {
namespace ral {

namespace {

template <typename T>
bool readFromBinaryStream(std::ifstream& fin, T& t) {
  if (!fin) return false;
  fin.read(reinterpret_cast<char*>(&t), sizeof(T));
  return static_cast<bool>(fin);
}

template <>
bool readFromBinaryStream(std::ifstream& fin, std::string& val) {
  if (!fin) return false;
  size_t bytes;
  if (!readFromBinaryStream(fin, bytes)) return false;
  std::string str(bytes, 0);
  fin.read(&str[0], bytes);
  if (!fin) return false;
  val = str;
  return true;
}

template <typename T>
bool writeToBinaryStream(std::ofstream& out, const T& t) {
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(&t), sizeof(T));
  return static_cast<bool>(out);
}

template <>
bool writeToBinaryStream(std::ofstream& out, const std::string& t) {
  if (!out) return false;
  if (!writeToBinaryStream(out, t.size())) return false;
  out.write(t.data(), t.size());
  return static_cast<bool>(out);
}

constexpr const uint64_t kMetadataFileMagicNumber = 0x1234567890ABCDEF;

}  // namespace

/* static */ std::unique_ptr<MetadataFile> MetadataFile::loadFromFile(
    const std::string& filename) {
  std::unique_ptr<MetadataFile> file(new MetadataFile);
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  uint64_t magicNumber = 0ull;
  if (!readFromBinaryStream(fin, magicNumber) ||
      magicNumber != kMetadataFileMagicNumber) {
    return nullptr;
  }
  fin.seekg(-static_cast<int64_t>(sizeof(size_t)), std::ios::end);
  size_t numConsts;
  if (!readFromBinaryStream(fin, numConsts)) return nullptr;
  fin.seekg(sizeof(kMetadataFileMagicNumber));
  for (size_t i = 0; i < numConsts; ++i) {
    bool isHost;
    std::string key, value;
    if (!readFromBinaryStream(fin, isHost)) return nullptr;
    if (!readFromBinaryStream(fin, key)) return nullptr;
    if (!readFromBinaryStream(fin, value)) return nullptr;
    if (isHost) {
      file->hostConstMap_.emplace(std::move(key), std::move(value));
    } else {
      file->deviceConstMap_.emplace(std::move(key), std::move(value));
    }
  }
  return file;
}

bool MetadataFile::getHostConstant(const std::string& name,
                                   const std::string*& data) {
  auto it = hostConstMap_.find(name);
  if (it == hostConstMap_.end()) return false;
  data = &it->second;
  return true;
}

bool MetadataFile::releaseHostConstant(const std::string& name) {
  auto it = hostConstMap_.find(name);
  if (it == hostConstMap_.end()) return false;
  hostConstMap_.erase(it);
  return true;
}

bool MetadataFile::getDeviceConstant(const std::string& name,
                                     const std::string*& data) {
  auto it = deviceConstMap_.find(name);
  if (it == deviceConstMap_.end()) return false;
  data = &it->second;
  return true;
}

bool MetadataFile::releaseDeviceConstant(const std::string& name) {
  auto it = deviceConstMap_.find(name);
  if (it == deviceConstMap_.end()) return false;
  deviceConstMap_.erase(it);
  return true;
}

MetadataFileEmitter::MetadataFileEmitter(const std::string& filename)
    : filename_(filename) {}

bool MetadataFileEmitter::emitHeader() {
  out_.open(filename_, std::ios::out | std::ios::binary);
  if (!writeToBinaryStream(out_, kMetadataFileMagicNumber)) return false;
  nextOffset_ += sizeof(kMetadataFileMagicNumber);
  return static_cast<bool>(out_);
}

bool MetadataFileEmitter::emitHostConstant(const std::string& name,
                                           const std::string& data) {
  // Returns false if there are duplicated keys.
  if (!hostConstMap_.emplace(name, nextOffset_).second) return false;
  // emit is_host?
  if (!writeToBinaryStream(out_, true)) return false;
  nextOffset_ += sizeof(bool);
  // emit the key
  if (!writeToBinaryStream(out_, name)) return false;
  nextOffset_ += sizeof(name.size()) + name.size();
  // emit the value
  if (!writeToBinaryStream(out_, data)) return false;
  nextOffset_ += sizeof(data.size()) + data.size();
  return true;
}

bool MetadataFileEmitter::emitDeviceConstant(const std::string& name,
                                             const std::string& data) {
  // Returns false if there are duplicated keys.
  if (!deviceConstMap_.emplace(name, nextOffset_).second) return false;
  if (!writeToBinaryStream(out_, false)) return false;
  nextOffset_ += sizeof(bool);
  // emit the key
  if (!writeToBinaryStream(out_, name)) return false;
  nextOffset_ += sizeof(name.size()) + name.size();
  // emit the value
  if (!writeToBinaryStream(out_, data)) return false;
  nextOffset_ += sizeof(data.size()) + data.size();
  return true;
}

bool MetadataFileEmitter::emitTailer() {
  size_t numConsts = hostConstMap_.size() + deviceConstMap_.size();
  if (!writeToBinaryStream(out_, numConsts)) return false;
  out_.close();
  return true;
}

}  // namespace ral
}  // namespace tao