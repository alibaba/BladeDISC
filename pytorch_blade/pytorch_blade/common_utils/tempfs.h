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

#pragma once

#include <cstdio>
#include <string>

#include "pytorch_blade/common_utils/macros.h"
namespace torch {
namespace blade {

class TempFile {
 public:
  TempFile(std::string prefix = "");
  ~TempFile();
  DISALLOW_COPY_AND_ASSIGN(TempFile);

  /// Write bytes content to temp file and return true on success.
  bool WriteBytesToFile(const std::string& bytes);

  /// Read byte content from temp file.
  std::string ReadBytesFromFile();

  /// Read string content from temp file..
  std::string ReadStringFromFile();

  /// Get the filename of the temp file.
  const std::string& GetFilename() const;

 private:
  std::string fname_;
  int fd_;
};
} // namespace blade
} // namespace torch
