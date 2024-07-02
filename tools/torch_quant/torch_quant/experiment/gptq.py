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

import contextlib
import logging
import math
from typing import Callable, List, Optional

import torch
from torch import nn
from torch_quant.observer import Observer
from torch_quant.quantizer import (DEFAULT_W_OB_CTR, Backend, Device,
                                   get_default_ctr)

LOGGER = logging.getLogger(__name__)


try:
    import transformers
    is_transformers_avail = True
except ModuleNotFoundError:
    LOGGER.warning("transformers is not installed, "
                   "so that gptq can not be applied to transformers.Conv1D")
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
    def __init__(self, layer_name, layer, observer_ctr):
        super().__init__()
        self.layer_name = layer_name
        self.layer = layer
        self.device = layer.weight.device
        self.gptq_observer = GPTQObserver(observer_ctr().to(self.device))
        columns = layer.weight.shape[1]
        self.columns = columns
        self.H = torch.zeros((columns, columns), device=self.device)
        self.nsamples = 0
        self.is_record = True

    def record_h(self, x):
        if self.is_record:
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
            logging.warning(f"Warning:  cannot do compression on layer {self.layer_name} because of inverse error")
            return

        if H.isnan().any():
            logging.warning(f"Warning:  cannot do compression on layer {self.layer_name} because of inverse error")
            return

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if is_transformer_conv1d(self.layer):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        del self.H
        del self.gptq_observer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GPTQModuleWrapper:
    def __init__(self, module_name: str, module: nn.Module, w_ob_ctr):
        self.all_layers = {}
        self.all_handles = []
        # module order in the whole network
        self.order = 0
        self.module_name = module_name

        def get_hook(layer_name):
            def record_hook(_, x):
                self.all_layers[layer_name].record_h(x[0])
            return record_hook

        for layer_name, layer in module.named_modules():
            if isinstance(layer, tuple(QUANT_LAYERS)):
                full_layer_name = f"{module_name}.{layer_name}" if layer_name else f"{module_name}"
                self.all_layers[full_layer_name] = GPTQLayerWrapper(full_layer_name, layer, w_ob_ctr)
                handle = layer.register_forward_pre_hook(get_hook(full_layer_name))
                self.all_handles.append(handle)

    def quant_module(self):
        for _, wrapper in self.all_layers.items():
            wrapper.quant_weight()

        for h in self.all_handles:
            h.remove()

    def set_order(self, idx):
        self.order = idx

    def get_order(self):
        return self.order

    def enable(self):
        for n, l in self.all_layers.items():
            l.is_record = True

    def disable(self):
        for n, l in self.all_layers.items():
            l.is_record = False


class GPTQuantizer:
    def __init__(self,
                 backend: Backend = Backend.DISC,
                 device: Device = Device.GPU,
                 block: Optional[List[type]] = None
                 ) -> None:
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
        def wrap_target_module(m, prefix=""):
            for name, child in m.named_children():
                new_prefix = f"{prefix}.{name}" if prefix else name
                if isinstance(child, tuple(self.block)):
                    self.all_module_wrappers[name] = GPTQModuleWrapper(new_prefix, child, w_ob_ctr)
                    LOGGER.debug(f"Calibrate module {new_prefix} as a whole block in GPTQ")
                else:
                    wrap_target_module(child, new_prefix)

        wrap_target_module(model)
        return model

    def quantize(self, model: nn.Module):
        for _, module_wrapper in self.all_module_wrappers.items():
            module_wrapper.quant_module()

        return model

    @property
    def calibration_iters(self):
        return len(self.all_module_wrappers)

    @contextlib.contextmanager
    def record_order(self):
        counter = 0
        record_handles = []
        orders = {}
        try:
            def get_record_order_hook(module_name):
                def record_hook(*args, **kwargs):
                    nonlocal counter
                    if module_name not in orders:
                        orders[module_name] = counter
                        counter += 1
                return record_hook

            for module_name, module_wrapper in self.all_module_wrappers.items():
                # disable the record
                for _, layer_wrapper in module_wrapper.all_layers.items():
                    layer_wrapper.is_record = False

                one_layer_wrapper_in_module = list(module_wrapper.all_layers.values())[0]
                handles = one_layer_wrapper_in_module.layer.register_forward_pre_hook(get_record_order_hook(module_name))
                record_handles.append(handles)
            yield
        except Exception as e:
            logging.warning(e)
        finally:
            for module_name, order in orders.items():
                self.all_module_wrappers[module_name].set_order(order)

            for h in record_handles:
                h.remove()

            for module_name, module_wrapper in self.all_module_wrappers.items():
                # disable the record
                for _, layer_wrapper in module_wrapper.all_layers.items():
                    layer_wrapper.is_record = True


    @contextlib.contextmanager
    def start_calib_iter(self, i):
        assert i < len(self.all_module_wrappers)
        target_module_wrapper = None
        try:
            for _, module_wrapper in self.all_module_wrappers.items():
                if module_wrapper.get_order() == i:
                    module_wrapper.enable()
                    target_module_wrapper = module_wrapper
                else:
                    module_wrapper.disable()
            yield
        finally:
            target_module_wrapper.quant_module()
