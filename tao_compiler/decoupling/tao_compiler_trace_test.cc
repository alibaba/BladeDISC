#include "tensorflow/compiler/decoupling/tao_compiler_trace.h"

#include <fstream>

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tao {

TEST(TaoCompilerTraceTest, Test) {
  std::string file;
  ASSERT_TRUE(tensorflow::Env::Default()->LocalTempFilename(&file));
  setenv("TAO_COMPILER_TRACE_DUMP_PATH", file.c_str(), 1);
  auto* ins = TaoCompilerTrace::Instance();
  ins->OnEvent("debug", "debug");
  ins->OnEvent("trace", "trace");
  ins->Shutdown();

  constexpr char sep = '\001';
  std::ifstream fin(file);
  std::string line;
  {
    ASSERT_TRUE(fin >> line);
    auto splits = str_util::Split(line, sep);
    ASSERT_EQ(splits.size(), 3);
    ASSERT_STREQ(splits[1].c_str(), "debug");
    ASSERT_STREQ(splits[2].c_str(), "debug");
  }
  {
    ASSERT_TRUE(fin >> line);
    auto splits = str_util::Split(line, sep);
    ASSERT_EQ(splits.size(), 3);
    ASSERT_STREQ(splits[1].c_str(), "trace");
    ASSERT_STREQ(splits[2].c_str(), "trace");
  }
  ASSERT_FALSE(fin >> line);
}

}  // namespace tao
}  // namespace tensorflow
