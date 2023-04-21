// Copyright 2023 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/script.h>
#include <cstdint>

namespace torch {
namespace blade {
std::tuple<at::Tensor, at::Tensor> torch_blade_attentionF(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_mask,
    const std::string& i_layout,
    const std::string& o_layout,
    bool is_causal,
    bool compute_logsumexp) {
  TORCH_CHECK(
      false,
      "nerver call this disc attentionF operator on eager mode, we need to lower this op to DISC custom call, please contact BladeDISC dev team!");
  Tensor out, logsumexp;
  return std::make_tuple(out, logsumexp);
}

TORCH_LIBRARY(disc, m) {
  m.def(
      "attentionF(Tensor query, Tensor key, Tensor value, Tensor attn_mask, str i_layout=\"BMHK\", str o_layout=\"BMHK\", bool is_causal=False, bool compute_logsumexp=False) -> (Tensor, Tensor)");
  m.impl("attentionF", &torch_blade_attentionF);
}

} //  namespace blade
} //  namespace torch