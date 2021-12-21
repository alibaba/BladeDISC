#ifndef __COMPILER_MLIR_RUNTIME_TEMPFS_H__
#define __COMPILER_MLIR_RUNTIME_TEMPFS_H__

#include <cstdio>
#include <string>

namespace torch {
namespace addons {

class TempFile {
 public:
  TempFile();
  ~TempFile();
  void WriteBytesToFile(const std::string& bytes);
  std::string ReadBytesFromFile();
  std::string ReadStringFromFile();
  std::string GetFilename();

 private:
  std::FILE* tmp_file_;
};
} // namespace addons
} // namespace torch
#endif //__COMPILER_MLIR_RUNTIME_TEMPFS_H__