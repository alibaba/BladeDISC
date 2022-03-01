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

import torch
import torch_blade


class wrapper(torch.nn.Module):
   def __init__(self, original_model):
        super().__init__()
        self.model=original_model

   def forward(
           self,
           inputs = None,
           ):
        generated_ids = self.model.generate(inputs)
        return generated_ids


def traceS2TAndSave(filename : str):
    from transformers import Speech2TextForConditionalGeneration
    model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
    model = model.eval()
    batch_size = 1
    sequence_length = 584
    feature_size = 80
    inputs = torch.rand(batch_size, sequence_length, feature_size)
    traced_model = torch.jit.trace(wrapper(model.cuda()), inputs.cuda(), strict = False)
    torch.jit.save(traced_model, filename)


def DISCOptimize(input_pt : str, output_pt_to_save : str):
    traced_model = torch.jit.load(input_pt)
    traced_model = traced_model.eval()

    batch_size = 5
    sequence_length = 584
    feature_size = 80
    input_features = torch.rand(batch_size, sequence_length, feature_size)

    torch_config = torch_blade.config.Config()
    torch_config.enable_mlir_amp = True # disable mix-precision
    torch_config.enable_force_to_cuda = True
    with torch.no_grad(), torch_config:
        # BladeDISC torch_blade optimize will return an optimized TorchScript
        optimized_ts = torch_blade.optimize(traced_model.cuda(), allow_tracing=True,
                                            model_inputs=input_features.cuda())

    torch.jit.save(optimized_ts, output_pt_to_save)
    optimized_ts = torch.jit.load(output_pt_to_save)

    output = optimized_ts(input_features.cuda())

    print(output)


if __name__ == '__main__':
    # A traced model is already maintained at:
    # http://zhengzhen.oss-cn-hangzhou-zmf.aliyuncs.com/model-data/Speech/PyTorch-S2T-Fairseq/traced-model.pt
    traceS2TAndSave('traced-model.pt')
    DISCOptimize('traced-model.pt', 'disc-opt.pt')
