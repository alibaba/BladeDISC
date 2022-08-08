// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>
#include <vector>

struct FoldAccGather : torch::CustomClassHolder {
  int64_t dim;
  int64_t size;
  FoldAccGather(int64_t init_dim, int64_t init_size) {
    dim = init_dim;
    size = init_size;
  }

  torch::Tensor forward(torch::Tensor x) {
    std::vector<int64_t> shape = x.sizes().vec();
    shape[dim] = shape[dim] * size;
    auto options = torch::TensorOptions()
                       .dtype(x.dtype())
                       .device(x.device())
                       .layout(torch::kStrided)
                       .requires_grad(false);
    return torch::randn(shape, options);
  }
};

struct FoldAccScatter : torch::CustomClassHolder {
  int64_t dim;
  int64_t size;
  FoldAccScatter(int64_t init_dim, int64_t init_size) {
    dim = init_dim;
    size = init_size;
  }

  torch::Tensor forward(torch::Tensor x) {
    std::vector<int64_t> shape = x.sizes().vec();
    shape[dim] = shape[dim] / size;
    auto options = torch::TensorOptions()
                       .dtype(x.dtype())
                       .device(x.device())
                       .layout(torch::kStrided)
                       .requires_grad(false);
    return torch::randn(shape, options);
  }
};

struct FoldAccAlltoAll : torch::CustomClassHolder {
  int64_t in_dim;
  int64_t out_dim;
  int64_t size;
  FoldAccAlltoAll(int64_t init_in_dim, int64_t init_out_dim,
                  int64_t init_size) {
    in_dim = init_in_dim;
    out_dim = init_out_dim;
    size = init_size;
  }

  torch::Tensor forward(torch::Tensor x) {
    std::vector<int64_t> shape = x.sizes().vec();
    shape[in_dim] = shape[in_dim] / size;
    shape[out_dim] = shape[out_dim] * size;
    auto options = torch::TensorOptions()
                       .dtype(x.dtype())
                       .device(x.device())
                       .layout(torch::kStrided)
                       .requires_grad(false);
    return torch::randn(shape, options);
  }
};

TORCH_LIBRARY(foldacc, m) {
  m.class_<FoldAccGather>("FoldAccGather")
      .def(torch::init<int64_t, int64_t>())
      .def("forward", &FoldAccGather::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<FoldAccGather>& self)
              -> std::vector<int64_t> {
            std::vector<int64_t> states({self->dim, self->size});
            return states;
          },
          [](std::vector<int64_t> state) -> c10::intrusive_ptr<FoldAccGather> {
            return c10::make_intrusive<FoldAccGather>(state[0], state[1]);
          });

  m.class_<FoldAccScatter>("FoldAccScatter")
      .def(torch::init<int64_t, int64_t>())
      .def("forward", &FoldAccScatter::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<FoldAccScatter>& self)
              -> std::vector<int64_t> {
            std::vector<int64_t> states({self->dim, self->size});
            return states;
          },
          [](std::vector<int64_t> state) -> c10::intrusive_ptr<FoldAccScatter> {
            return c10::make_intrusive<FoldAccScatter>(state[0], state[1]);
          });

  m.class_<FoldAccAlltoAll>("FoldAccAlltoAll")
      .def(torch::init<int64_t, int64_t, int64_t>())
      .def("forward", &FoldAccAlltoAll::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<FoldAccAlltoAll>& self)
              -> std::vector<int64_t> {
            std::vector<int64_t> states(
                {self->in_dim, self->out_dim, self->size});
            return states;
          },
          [](std::vector<int64_t> state)
              -> c10::intrusive_ptr<FoldAccAlltoAll> {
            return c10::make_intrusive<FoldAccAlltoAll>(state[0], state[1],
                                                        state[2]);
          });
}