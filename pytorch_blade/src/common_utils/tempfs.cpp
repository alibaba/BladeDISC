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

#include "common_utils/tempfs.h"

#include <fstream>
#include <iostream>
#include <string>
#include "common_utils/logging.h"

namespace torch {
namespace blade {
TempFile::~TempFile() {
  if (tmp_file_ != nullptr) {
    std::fclose(tmp_file_);
  }
}

TempFile::TempFile() : tmp_file_(std::tmpfile()) {
  CHECK_NOTNULL(tmp_file_);
}

bool TempFile::WriteBytesToFile(const std::string& bytes) {
  auto sz = std::fwrite(bytes.data(), sizeof(char), bytes.size(), tmp_file_);
  if (sz != bytes.length()) {
    LOG(ERROR) << "Failed to write content to temp file: " << GetFilename()
               << ", error: " << strerror(errno);
    return false;
  }
  auto ret = std::fflush(tmp_file_);
  if (ret != 0) {
    LOG(ERROR) << "Failed to flush content to temp file: " << GetFilename()
               << ", error: " << strerror(errno);
    return false;
  }
  return true;
}

std::string TempFile::GetFilename() {
  char fname[FILENAME_MAX] = {0};
  sprintf(fname, "/proc/self/fd/%d", fileno(tmp_file_));
  return fname;
}

std::string TempFile::ReadBytesFromFile() {
  auto filename = GetFilename();
  std::ifstream infile(filename, std::ios::binary);
  std::string str(
      (std::istreambuf_iterator<char>(infile)),
      std::istreambuf_iterator<char>());
  return str;
}

std::string TempFile::ReadStringFromFile() {
  auto filename = GetFilename();
  std::ifstream infile(filename);
  std::string str(
      (std::istreambuf_iterator<char>(infile)),
      std::istreambuf_iterator<char>());
  return str;
}
} // namespace blade
} // namespace torch
