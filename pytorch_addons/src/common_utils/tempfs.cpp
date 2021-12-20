#include "common_utils/tempfs.h"

#include <iostream>
#include "common_utils/logging.h"

namespace torch {
namespace addons {
TempFile::~TempFile() {
  if (tmp_file_ != nullptr) {
    std::fclose(tmp_file_);
  }
}

TempFile::TempFile() : tmp_file_(std::tmpfile()) {
  CHECK_NOTNULL(tmp_file_);
}

void TempFile::WriteBytesToFile(const std::string& bytes) {
  std::fwrite(bytes.data(), sizeof(char), bytes.size(), tmp_file_);
  std::fflush(tmp_file_);
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
} // namespace addons
} // namespace torch
