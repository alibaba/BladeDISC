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

import json
import logging
import os
from os import PathLike
from tempfile import TemporaryDirectory
from typing import Union

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.pipelines.pipeline_utils import LOADABLE_CLASSES
from transformers import CLIPTextModel, PreTrainedModel

LOGGER = logging.getLogger(__name__)


class OptModel:
    original_class = None
    model_file = 'model.jit'

    def __init__(self, opt_model: torch.jit.ScriptModule):
        self.opt_model = opt_model

    def save_pretrained(self, save_directory: Union[str, PathLike], **kwargs):
        torch.jit.save(self.opt_model, os.path.join(
            save_directory, self.model_file))

    @classmethod
    def gen_example_input(cls):
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, cached_dir: Union[str, PathLike], **kwargs):
        if os.path.isfile(os.path.join(cached_dir, cls.model_file)):
            return cls.from_opt(cached_dir)
        else:
            return cls.from_original(cached_dir, **kwargs)

    @classmethod
    def from_opt(cls, cached_dir: Union[str, PathLike]):
        opt_model = torch.jit.load(os.path.join(cached_dir, cls.model_file))
        return cls(opt_model)

    @classmethod
    def from_original(cls, cached_dir: Union[str, PathLike], **kwargs):
        if issubclass(cls.original_class, PreTrainedModel):
            kwargs['torchscript'] = True
        # TODO(litan.ls): use load method from LOADABLE_CLASSES
        original = cls.original_class.from_pretrained(cached_dir, **kwargs)

        example_inputs = cls.gen_example_input()
        traced = torch.jit.trace(original.eval(), example_inputs)
        # TODO(litan.ls): call blade optimize
        return cls(traced)


class BladeCLIPTextModel(OptModel):
    original_class = CLIPTextModel

    @classmethod
    def gen_example_input(cls):
        return torch.randint(1, 999, (1, 10), dtype=torch.int64)

    def __call__(self, *args):
        # TODO(litan.ls): wrapper output as original model
        return self.opt_model(*args)


class BladeUNet2DConditionModel(OptModel):
    original_class = UNet2DConditionModel

    @classmethod
    def gen_example_input(cls):
        # TODO(litan.ls): support gen input from pipeline config
        return (
            torch.randn((1, 4, 64, 64), dtype=torch.half),
            torch.tensor(2, dtype=torch.int64),
            torch.randn((1, 10, 768), dtype=torch.half),
        )

    def forward(self, *args):
        return UNet2DConditionOutput(self.opt_model(*args))


# TODO(litan.ls): support more models
_MODEL_MAPPING = {
    'text_encoder': (['transformers', 'CLIPTextModel'], ['blade_adapter', 'BladeCLIPTextModel']),
    'unet': (['diffusers', 'UNet2DConditionModel'], ['blade_adapter', 'BladeUNet2DConditionModel']),
}


class BladeStableDiffusionPipeline(StableDiffusionPipeline):
    @classmethod
    def overwrite_config(cls, input_cached_dir: Union[str, PathLike], output_cached_dir: Union[str, PathLike]):
        config_dict = cls.load_config(input_cached_dir)
        for k, (src_model, dst_model) in _MODEL_MAPPING.items():
            if k not in config_dict:
                LOGGER.warn(f'{k} model not found in pipeline config.')
            elif config_dict[k] != src_model:
                LOGGER.warn(f'Cannot overwrite {k} model type {src_model}')
            else:
                config_dict[k] = dst_model
        for dirpath, _, filenames in os.walk(input_cached_dir):
            relpath = os.path.relpath(dirpath, input_cached_dir)
            os.makedirs(os.path.join(
                output_cached_dir, relpath), exist_ok=True)
            for f in filenames:
                os.symlink(os.path.abspath(os.path.join(dirpath, f)),
                           os.path.join(output_cached_dir, relpath, f))
        config_path = os.path.join(output_cached_dir, cls.config_name)
        os.unlink(config_path)
        with open(config_path, 'w') as config_file:
            config_file.write(json.dumps(config_dict, indent=2))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, PathLike]):
        if not os.path.isdir(pretrained_model_name_or_path):
            raise NotImplementedError('Support snapshot download')
        else:
            cached_dir = pretrained_model_name_or_path

        with TemporaryDirectory() as tmpdir:
            cls.overwrite_config(cached_dir, tmpdir)
            LOADABLE_CLASSES['blade_adapter'] = {
                "OptModel": ["save_pretrained", "from_pretrained"]}
            return super().from_pretrained(tmpdir)
