//===- ral_metadata.h ----------------------===//
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

#ifndef RAL_RAL_METADATA_H_
#define RAL_RAL_METADATA_H_

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace tao {
namespace ral {

// A metadata file is generated during compilation and is used to aid the
// execution of the compiled binary. it usually includes:
// - (maybe large) const values, their shape and checksum
// - other related information used to aid the execution of the compiled binary.
class MetadataFile {
 public:
  // Loads the metadata file named `filename`.
  // Return nullptr if failed, otherwise a new MetadataFile instance.
  static std::unique_ptr<MetadataFile> loadFromFile(
      const std::string& filename);

  // Returns true if there is a host const named `name` and `data` is set to the
  // corresponding value. Otherwise return false and `data` is leaved untouched.
  bool getHostConstant(const std::string& name, const std::string*& data);

  // Returns true if there is a host constant named `name` and frees the memory
  // of such constant. Following `getHostConstant` with same key will return
  // false. Returns false if no such host constant.
  bool releaseHostConstant(const std::string& name);

  // Returns true if there is a device const named `name` and `data` is set to
  // the corresponding value. Otherwise return false and `data` is leaved
  // untouched.
  bool getDeviceConstant(const std::string& name, const std::string*& data);

  // Returns true if there is a device constant named `name` and frees the
  // memory of such constant. Following `getDeviceConstant` with same key will
  // return false. Returns false if no such device constant.
  bool releaseDeviceConstant(const std::string& name);

 private:
  // Disallows new a MetadataFile directly.
  explicit MetadataFile() = default;

  // Disables copy and assignment methods.
  MetadataFile(const MetadataFile& other) = delete;
  MetadataFile& operator=(const MetadataFile& other) = delete;

  std::unordered_map<std::string, std::string> hostConstMap_;
  std::unordered_map<std::string, std::string> deviceConstMap_;
};

// A helper class which is used to emit metadata file during compilation.
class MetadataFileEmitter {
 public:
  explicit MetadataFileEmitter(const std::string& filename);

  // Disables copy and assignment methods.
  MetadataFileEmitter(const MetadataFileEmitter& other) = delete;
  MetadataFileEmitter& operator=(const MetadataFileEmitter& other) = delete;

  // Emits the header of the metadata file.
  // This method should be called only once and before all other emit-like
  // methods. Returns false if encountering any errors, otherwise true.
  bool emitHeader();

  // Returns the number of host consts that have been emitted.
  int getNumHostConstantEmitted() { return hostConstMap_.size(); }

  // Emits a new host constant with key `name` and value `data`.
  bool emitHostConstant(const std::string& name, const std::string& data);

  // Returns the number of device consts that have been emitted.
  int getNumDeviceConstantEmitted() { return deviceConstMap_.size(); }

  // Emits a new device constant with key `name` and value `data.
  bool emitDeviceConstant(const std::string& name, const std::string& data);

  // Emits the tailer of the metadata file.
  // This method should be called only once and after all other emit-like
  // methods. Returns false if encountering any errors, otherwise true.
  bool emitTailer();

 private:
  std::string filename_;
  std::ofstream out_;
  uint64_t nextOffset_ = 0;
  // map key -> offset of beginning the const inside the metadata file.
  std::unordered_map<std::string, uint64_t> hostConstMap_;
  std::unordered_map<std::string, uint64_t> deviceConstMap_;
};

}  // namespace ral
}  // namespace tao

#endif  // RAL_RAL_METADATA_H_