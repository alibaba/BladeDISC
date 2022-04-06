#include <gtest/gtest.h>

#include <common_utils/tempfs.h>

using torch::blade::TempFile;

TEST(Tempfs, TestNormal) {
  TempFile f;
  std::string payload("hello, I'am the payload.");
  ASSERT_TRUE(f.WriteBytesToFile(payload));

  auto loaded_bytes = f.ReadBytesFromFile();
  ASSERT_EQ(payload, loaded_bytes);

  auto loaded_str = f.ReadStringFromFile();
  ASSERT_EQ(payload, loaded_str);

  ASSERT_FALSE(f.GetFilename().empty());
}
