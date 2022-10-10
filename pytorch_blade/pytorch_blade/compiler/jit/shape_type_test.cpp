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

#include <gtest/gtest.h>

#include <vector>

#include "pytorch_blade/compiler/jit/shape_type_spec.h"

using namespace torch::blade;
TEST(ShapeTypeTest, ShapeTypeString) {
  ShapeType shape_type;
  shape_type.type = at::ScalarType::Int;
  shape_type.shape = {1, 3, 224, 224};

  std::string serial_str = shape_type.Serialize();
  ASSERT_EQ(serial_str, "Int(1:1,3:3,224:224,224:224,)");
  ASSERT_EQ(shape_type, ShapeType::Deserialize(serial_str));

  shape_type.shape = {};
  serial_str = shape_type.Serialize();
  ASSERT_EQ(serial_str, "Int()");
  ASSERT_EQ(shape_type, ShapeType::Deserialize(serial_str));

  serial_str = "Flout(1:1,3:3,224:224,224:224,)";
  ASSERT_THROW(ShapeType::Deserialize(serial_str), c10::Error);

  serial_str = "Int(1:1, 3:3, 224:224, 224:224,)";
  ASSERT_THROW(ShapeType::Deserialize(serial_str), c10::Error);
}

TEST(ShapeTypeSpecTest, ShapeTypeSpecString) {
  at::List<at::Tensor> list;
  auto option = torch::dtype(torch::kInt8);
  list.push_back(torch::ones({1, 3, 2, 2}).to(option));
  option = torch::dtype(torch::kDouble);
  list.push_back(torch::ones({2, 2}).to(option));
  list.push_back(torch::randn({2, 2}));
  list.push_back(torch::ones({}));

  ShapeTypeSpec spec = ShapeTypeSpec::GetShapeTypeSpec(list);
  std::string serial_str = spec.Serialize();
  ASSERT_EQ(
      serial_str,
      "Char(1:1,3:3,2:2,2:2,);Double(2:2,2:2,);Float(2:2,2:2,);Float();");
  ASSERT_EQ(spec, ShapeTypeSpec::Deserialize(serial_str));
  ASSERT_THROW(ShapeTypeSpec::Deserialize(serial_str + ";"), c10::Error);
  ASSERT_THROW(
      ShapeTypeSpec::Deserialize(serial_str + "Float(1,);"), c10::Error);
  serial_str += "Int(1:1,);";
  ASSERT_FALSE(spec == ShapeTypeSpec::Deserialize(serial_str));
}
