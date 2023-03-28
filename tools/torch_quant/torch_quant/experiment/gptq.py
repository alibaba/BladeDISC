# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from typing import Callable, List, Optional

import torch
from torch import nn
from torch_quant.module import ModuleFilter
from torch_quant.observer import Observer
from torch_quant.quantizer import DEFAULT_W_OB_CTR, Backend, Device, get_default_ctr

LOGGER = logging.getLogger(__name__)


try:
    import transformers
    is_transformers_avail = True
except ModuleNotFoundError:
    LOGGER.warning("transformers is not installed, so that gptq can not be applied to transformers.Conv1D")
    is_transformers_avail = False


# TODO: For models that can not run the gptq process within single GPU
# we should support cpu offload.

QUANT_LAYERS = [nn.Linear, nn.Conv2d]
if is_transformers_avail:
    QUANT_LAYERS.append(transformers.Conv1D)


def is_transformer_conv1d(layer):
    return is_transformers_avail and isinstance(layer, transformers.Conv1D)


class GPTQObserver:
    def __init__(self, observer: Observer):
        self.observer = observer
        self.scales = []
        self.zero_points = []

    def find_quant_info(self, x):
        self.observer.set_mode(observe=True, fake_quant=False)
        self.observer(x)
        self.scales.append(self.observer.scale)
        self.zero_points.append(self.observer.zero_point)
        # todo: reset the observer

    def fake_quant(self, x):
        self.observer.set_mode(observe=False, fake_quant=True)
        x = self.observer(x)
        return x


class GPTQLayerWrapper:
    def __init__(self, layer, observer):
        super().__init__()
        self.layer = layer
        self.gptq_observer = GPTQObserver(observer)
        self.device = layer.weight.device
        columns = layer.weight.shape[1]
        self.columns = columns
        self.H = torch.zeros((columns, columns), device=self.device)
        self.nsamples = 0

    def record_h(self, x):
        x = x.detach().clone()
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        if isinstance(self.layer, nn.Linear) or is_transformer_conv1d(self.layer):
            if len(x.shape) == 3:
                x = x.reshape((-1, x.shape[-1]))
            x = x.t()

        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            x = unfold(x)
            x = x.permute([1, 0, 2])
            x = x.flatten(1)

        self.H *= self.nsamples / (self.nsamples + batch)
        self.nsamples += batch
        x = math.sqrt(2 / self.nsamples) * x.float()
        self.H += x.matmul(x.t())

    def quant_weight(self, blocksize=128, percdamp=.01, groupsize=-1):
        weight = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if is_transformer_conv1d(self.layer):
            weight = weight.t()
        weight = weight.float()

        if groupsize == -1:
            self.gptq_observer.find_quant_info(weight)

        H = self.H
        # del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        weight[:, dead] = 0

        losses = torch.zeros_like(weight)
        Q = torch.zeros_like(weight)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
        except Exception:
            # TODO: should handle this situation
            logging.warning("Warning:  cannot do compression for inverse error")

        if H.isnan().any():
            # TODO: should handle this situation
            logging.warning("Warning:  cannot do compression for inverse error")

        hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            w1 = weight[:, i1:i2].clone()
            q1 = torch.zeros_like(w1)
            total_err = torch.zeros_like(w1)
            losses1 = torch.zeros_like(w1)
            hinv1 = hinv[i1:i2, i1:i2]

            for i in range(count):
                w = w1[:, i]
                d = hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.gptq_observer.find_quant_info(weight[:, (i1 + i):(i1 + i + groupsize)])
                q = self.gptq_observer.fake_quant(w.unsqueeze(1)).flatten()

                q1[:, i] = q
                losses1[:, i] = (w - q) ** 2 / d ** 2
                err = (w - q) / d
                w1[:, i:] -= err.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                total_err[:, i] = err

            Q[:, i1:i2] = q1
            losses[:, i1:i2] = losses1 / 2

            weight[:, i2:] -= total_err.matmul(hinv[i1:i2, i2:])

        # torch.cuda.synchronize()

        if is_transformer_conv1d(self.layer):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


class GPTQModuleWrapper:
    def __init__(self, module: nn.Module, w_ob_ctr):
        self.all_layers = {}
        self.all_handles = []

        def get_hook(layer_name):
            def record_hook(_, x):
                self.all_layers[layer_name].record_h(x[0])
            return record_hook

        for name, layer in module.named_modules():
            if isinstance(layer, tuple(QUANT_LAYERS)):
                self.all_layers[name] = GPTQLayerWrapper(layer, w_ob_ctr())
                handle = layer.register_forward_pre_hook(get_hook(name))
                self.all_handles.append(handle)

    def quant_module(self):
        for _, wrapper in self.all_layers.items():
            wrapper.quant_weight()

        for h in self.all_handles:
            h.remove()


class GPTQuantizer:
    def __init__(self, module_filter: Optional[ModuleFilter] = None,
                 backend: Backend = Backend.DISC,
                 device: Device = Device.GPU,
                 block: Optional[List[nn.Module]] = None
                 ) -> None:
        self.module_filter = module_filter
        self.backend = backend
        self.device = device
        self.all_module_wrappers = {}
        self.block = block or QUANT_LAYERS

    def calib(self, model: nn.Module,
              w_ob_ctr: Optional[Callable[..., Observer]] = None):
        default_w_ob_ctr = get_default_ctr(DEFAULT_W_OB_CTR, self.device, self.backend)
        w_ob_ctr = w_ob_ctr or default_w_ob_ctr
        # GPTQ only quantize the weight of LLMs, so there is no need to
        # convert it to fx module
        for name, module in model.named_modules():
            if isinstance(module, tuple(self.block)):
                self.all_module_wrappers[name] = GPTQModuleWrapper(module, w_ob_ctr)

        return model

    def quantize(self, model: nn.Module):
        for _, module_wrapper in self.all_module_wrappers.items():
            module_wrapper.quant_module()

        return model


if __name__ == "__main__":
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2048, 2048)

        def forward(self, x):
            return self.linear(x)

    model = MyModel()
    ss = dict(model.named_modules())
    quantizer = GPTQuantizer()
    calib_model = quantizer.calib(model)
    dummy = torch.randn(1, 2048, 2048)
    calib_model(dummy)
    quant_model = quantizer.quantize(model)
