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

struct PaiGather : torch::CustomClassHolder {
  int64_t dim;
  int64_t size;
  PaiGather(int64_t init_dim, int64_t init_size) {
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

struct PaiScatter : torch::CustomClassHolder {
  int64_t dim;
  int64_t size;
  PaiScatter(int64_t init_dim, int64_t init_size) {
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

struct PaiAlltoAll : torch::CustomClassHolder {
  int64_t in_dim;
  int64_t out_dim;
  int64_t size;
  PaiAlltoAll(int64_t init_in_dim, int64_t init_out_dim, int64_t init_size) {
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

TORCH_LIBRARY(paifold, m) {
  m.class_<PaiGather>("PaiGather")
      .def(torch::init<int64_t, int64_t>())
      .def("forward", &PaiGather::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<PaiGather>& self)
              -> std::vector<int64_t> {
            std::vector<int64_t> states({self->dim, self->size});
            return states;
          },
          [](std::vector<int64_t> state) -> c10::intrusive_ptr<PaiGather> {
            return c10::make_intrusive<PaiGather>(state[0], state[1]);
          });

  m.class_<PaiScatter>("PaiScatter")
      .def(torch::init<int64_t, int64_t>())
      .def("forward", &PaiScatter::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<PaiScatter>& self)
              -> std::vector<int64_t> {
              std::vector<int64_t> states({self->dim, self->size});
            return states;
          },
          [](std::vector<int64_t> state) -> c10::intrusive_ptr<PaiScatter> {
            return c10::make_intrusive<PaiScatter>(state[0], state[1]);
          });

  m.class_<PaiAlltoAll>("PaiAlltoAll")
      .def(torch::init<int64_t, int64_t, int64_t>())
      .def("forward", &PaiAlltoAll::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<PaiAlltoAll>& self)
              -> std::vector<int64_t> {
            std::vector<int64_t> states(
                {self->in_dim, self->out_dim, self->size});
            return states;
          },
          [](std::vector<int64_t> state) -> c10::intrusive_ptr<PaiAlltoAll> {
            return c10::make_intrusive<PaiAlltoAll>(state[0], state[1],
                                                    state[2]);
          });
}