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



import decorator
import contextlib

def wrap_decorator(decorator_func):
    @decorator.decorator
    def __impl__(func, *args, **kwargs):
        wrapped_func = decorator_func(func)
        return wrapped_func(*args, **kwargs)

    return __impl__

signature_safe_contextmanager = wrap_decorator(contextlib.contextmanager)

class DiscContext:
    def __init__(self):
        self._fx_gms : List[torch.fx.GraphModule] = []
        pass
    @property
    def fx_gms(self):
        return self._fx_gms

_default_disc_context = DiscContext()

def default_disc_context():
    return _default_disc_context

def switch_disc_context(ctx):
    global _default_disc_context
    prev_context = _default_disc_context
    _default_disc_context = ctx
    return prev_context

@signature_safe_contextmanager
def disc_context_guard(ctx):
    """
    Examples:
        import torch
        import torch_blade

        ctx = torch_blade.default_disc_context()
        with torch_blade.disc_context_guard(ctx):
            # do something
            net = torch.compile(...)(net)
            net(data, label)
            
            # print the graph for debug
            print(fx.fx_gms[0].graph)

    """
    ctx = switch_disc_context(ctx)
    try:
        yield
    finally:
        switch_disc_context(ctx)