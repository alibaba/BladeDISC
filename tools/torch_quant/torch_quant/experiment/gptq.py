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

    def record(self, x):
        x = x.detach().clone()
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        if isinstance(self.layer, nn.Linear) or (is_transformers_avail and isinstance(self.layer, transformers.Conv1D)):
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

    def quant(self, blocksize=128, percdamp=.01, groupsize=-1):
        weight = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if is_transformers_avail and isinstance(self.layer, transformers.Conv1D):
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
        mask = torch.ones_like(weight)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
        except Exception:
            # TODO: should handle this situation
            logging.warning(f"Warning:  cannot do compression for inverse error")

        if H.isnan().any():
            # TODO: should handle this situation
            logging.warning(f"Warning:  cannot do compression for inverse error")

        hinv = H
        hinv_diag = torch.diag(hinv)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            w1 = weight[:, i1:i2].clone()
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
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
                err1 = (w - q) / d
                w1[:, i:] -= err1.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                err1[:, i] = err1

            Q[:, i1:i2] = q1
            losses[:, i1:i2] = losses1 / 2

            weight[:, i2:] -= err1.matmul(hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


class GPTQModuleWrapper:
    def __init__(self, module: nn.Module, w_ob_ctr):
        self.all_layers = {}
        self.all_handles = []
        def get_hook(name):
            def record(_, x):
                self.all_layers[name].record(x[0])
            return record

        for name, layer in module.named_modules():
            if isinstance(layer, tuple(QUANT_LAYERS)):
                self.all_layers[name] = GPTQLayerWrapper(layer, w_ob_ctr())
                handle = layer.register_forward_pre_hook(get_hook(name))
                self.all_handles.append(handle)


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
        pass



if __name__ == "__main__":
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4)

        def forward(self, x):
            return self.linear(x)

    model = MyModel()
    ss = dict(model.named_modules())
    quantizer = GPTQuantizer()
    calib_model = quantizer.calib(model)
    dummy = torch.randn(1, 3)
    calib_model(dummy)
    quant_model = quantizer.quantize(model)
