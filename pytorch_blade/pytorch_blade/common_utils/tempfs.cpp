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

#include "pytorch_blade/common_utils/tempfs.h"

#include <unistd.h>

#include <fstream>
#include <string>

#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/common_utils/utils.h"

namespace torch {
namespace blade {

namespace {
// This function is copied from c10/util/tempfile.h, so it follows to these
// temperary directory env variables, too.
std::vector<char> make_filename(std::string name_prefix) {
  // The filename argument to `mkstemp` needs "XXXXXX" at the end according to
  // http://pubs.opengroup.org/onlinepubs/009695399/functions/mkstemp.html
  static const std::string kRandomPattern = "XXXXXX";

  // We see if any of these environment variables is set and use their value, or
  // else default the temporary directory to `/tmp`.
  static const char* env_variables[] = {"TMPDIR", "TMP", "TEMP", "TEMPDIR"};

  std::string tmp_directory = "/tmp";
  for (const char* variable : env_variables) {
    auto path = env::ReadStringFromEnvVar(variable, "");
    if (!path.empty()) {
      tmp_directory = path;
      break;
    }
  }

  std::vector<char> filename;
  filename.reserve(
      tmp_directory.size() + name_prefix.size() + kRandomPattern.size() + 2);

  filename.insert(filename.end(), tmp_directory.begin(), tmp_directory.end());
  filename.push_back('/');
  filename.insert(filename.end(), name_prefix.begin(), name_prefix.end());
  filename.insert(filename.end(), kRandomPattern.begin(), kRandomPattern.end());
  filename.push_back('\0');

  return filename;
}
} // namespace

TempFile::TempFile(std::string prefix) : fname_(""), fd_(-1) {
  auto fname = make_filename(prefix);
  fd_ = mkstemp(fname.data());
  fname_ = std::string(fname.data());
  if (fd_ == -1) {
    LOG(FATAL) << "Error generating temporary file, file name: " << fname_
               << ", error: " << std::strerror(errno);
  }
}

TempFile::~TempFile() {
  if (!fname_.empty()) {
    ::unlink(fname_.c_str());
  }
  if (fd_ > 0) {
    ::close(fd_);
  }
}

bool TempFile::WriteBytesToFile(const std::string& bytes) {
  ssize_t left_len = bytes.length();
  const char* data = bytes.data();
  errno = 0;
  while (left_len > 0) {
    auto sz = ::write(fd_, data, left_len);
    if (sz <= 0) {
      if (errno != EINTR && errno != EAGAIN) {
        LOG(ERROR) << "Failed to write content to temp file: " << GetFilename()
                   << ", error: " << strerror(errno);
        return false;
      }
      errno = 0;
      continue;
    }
    left_len -= sz;
    data += sz;
  }

  return true;
}

const std::string& TempFile::GetFilename() const {
  return fname_;
}

std::string TempFile::ReadBytesFromFile() {
  std::ifstream infile(fname_, std::ios::binary);
  std::string str(
      (std::istreambuf_iterator<char>(infile)),
      std::istreambuf_iterator<char>());
  return str;
}

std::string TempFile::ReadStringFromFile() {
  std::ifstream infile(fname_);
  std::string str(
      (std::istreambuf_iterator<char>(infile)),
      std::istreambuf_iterator<char>());
  return str;
}
} // namespace blade
} // namespace torch
