# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from torch_blade._torch_blade._neural_engine import *
    from torch_blade.config import OptPipelines
    from torch_blade.neural_engine.neural_engine_optimization import \
        optimize_neural_engine
    _NEURAL_ENGINE_NAME = backend_name()
    OptPipelines.register_pipeline(_NEURAL_ENGINE_NAME, optimize_neural_engine)
    _is_available = True
except ImportError:
    _is_available = False
    _NEURAL_ENGINE_NAME = "NeuralEngine"


def is_available():
    return _is_available
