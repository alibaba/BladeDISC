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

import logging
from copy import deepcopy
from inspect import signature
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import torch
import torch_blade
from transformers import (AutoConfig, AutoFeatureExtractor, AutoTokenizer,
                          FeatureExtractionMixin, Pipeline, PretrainedConfig,
                          PreTrainedModel, PreTrainedTokenizer, ProcessorMixin)
from transformers import pipeline as hf_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.models.auto.auto_factory import _get_model_class
from transformers.onnx.features import FeaturesManager
from transformers.onnx.utils import get_preprocessor
from transformers.pipelines import (check_task, get_default_model_and_revision,
                                    get_task, infer_framework_load_model)
from transformers.utils import TensorType
from transformers.utils.generic import ModelOutput

LOGGER = logging.getLogger(__name__)

Preprocessor = Union[PreTrainedTokenizer,
                     FeatureExtractionMixin, ProcessorMixin]


class ModelInfo(NamedTuple):
    task: str
    id_or_path: str
    config: PretrainedConfig
    targeted_task: Dict[str, Any]
    model_class: Type[PreTrainedModel]
    dummy_inputs: Dict[str, torch.Tensor]
    input_order: List[str]
    default_kwargs: Dict[str, Any]
    output_names: List[str]


# pipeline task to model task mapping
TASK_MAP = {
    'audio-classification': 'sequence-classification',  # ?
    'automatic-speech-recognition': 'speech2seq-lm',
    'feature-extraction': 'default',  # ?
    'text-classification': 'sequence-classification',
    'token-classification': 'token-classification',
    'question-answering': 'question-answering',
    # 'table-question-answering': '',
    # 'visual-question-answering': '',
    # 'document-question-answering': '',
    'fill-mask': 'masked-lm',
    'summarization': 'seq2seq-lm',
    'translation': 'seq2seq-lm',
    'text2text-generation': 'seq2seq-lm',
    'text-generation': 'causal-lm',
    'zero-shot-classification': 'sequence-classification',
    'zero-shot-image-classification': 'image-classification',  # ?
    'conversational': 'seq2seq-lm',  # ?
    'image-classification': 'image-classification',
    'image-segmentation': 'image-segmentation',
    'image-to-text': 'vision2seq-lm',
    'object-detection': 'object-detection',
    'zero-shot-object-detection': 'object-detection',  # ?
    # 'depth-estimation': '',
}


def load_model_info(task: Optional[str] = None, id_or_path: Optional[str] = None,
                    config: Optional[PretrainedConfig] = None,
                    model_class: Optional[Type[PreTrainedModel]] = None,
                    preprocessor: Optional[Preprocessor] = None) -> ModelInfo:
    if task is None and id_or_path is None:
        raise ValueError('Must specify task or model id/model path.')
    if task is None:
        task = get_task(id_or_path)
    # TODO(litan.ls): version compat for check_task
    task, targeted_task, task_options = check_task(task)
    if task not in TASK_MAP:
        raise ValueError(f'Unsupported pipeline task: {task}')

    if id_or_path is None:
        if config is None:
            id_or_path, _ = get_default_model_and_revision(
                targeted_task, framework='pt', task_options=task_options)
        else:
            id_or_path = config._name_or_path

    if config is None:
        config = AutoConfig.from_pretrained(id_or_path, _from_pipeline=task)

    model_features = FeaturesManager.get_supported_features_for_model_type(
        config.model_type.replace("_", "-"))
    if TASK_MAP[task] not in model_features:
        raise KeyError(
            f'{TASK_MAP[task]} is not supported for {config.model_type}')
    onnx_config_cls = model_features[TASK_MAP[task]]
    # deepcopy config to avoid being modified
    onnx_config = onnx_config_cls(deepcopy(config), task=TASK_MAP[task])
    if preprocessor is None:
        preprocessor = get_preprocessor(id_or_path)
    dummy_inputs = onnx_config.generate_dummy_inputs(
        preprocessor, framework=TensorType.PYTORCH)

    if model_class is None:
        # TODO(litan.ls): try load pretrained model to determine model class
        # TODO(litan.ls): refine the over-simplified model class deduction here
        model_class = _get_model_class(
            config, targeted_task['pt'][0]._model_mapping)
        LOGGER.debug(f'deduced model class: {model_class.__name__}')
    sig = signature(model_class.forward)
    all_keys = list(sig.parameters.keys())
    if all_keys[0] == 'self':
        all_keys = all_keys[1:]
    # TODO(litan.ls): handle default_args
    default_kwargs = {}
    input_order = []
    for k in all_keys:
        if k in dummy_inputs:
            default_kwargs[k] = Ellipsis
            input_order.append(k)
        else:
            # TODO(litan.ls): handle empty default
            default_kwargs[k] = sig.parameters[k].default

    return ModelInfo(task=task, id_or_path=id_or_path, config=config,
                     targeted_task=targeted_task, model_class=model_class,
                     dummy_inputs=dummy_inputs, input_order=input_order,
                     default_kwargs=default_kwargs,
                     output_names=list(onnx_config.outputs.keys()))


class Tracable(torch.nn.Module):
    def __init__(self, model: torch.nn.Module,
                 default_args: List[Any], default_kwargs: Dict[str, Any],
                 amp: bool = False) -> None:
        super().__init__()
        self.model = model
        self.default_args = default_args
        self.default_kwargs = default_kwargs
        self.amp = amp
        self.device_type = next(model.parameters()).device.type

    def _create_model_args(self, *args):
        model_args = []
        i = 0
        for x in self.default_args:
            if x is Ellipsis:
                model_args.append(args[i])
                i += 1
            else:
                model_args.append(x)
        model_kwargs = {}
        for k, v in self.default_kwargs.items():
            if v is Ellipsis:
                model_kwargs[k] = args[i]
                i += 1
            else:
                model_kwargs[k] = v
        return model_args, model_kwargs

    def forward(self, *args):
        model_args, model_kwargs = self._create_model_args(*args)
        with torch.autocast(self.device_type, enabled=self.amp):
            return self.model(*model_args, **model_kwargs)


def _kwargs_to_args(input_order, **kwargs) -> Tuple:
    return tuple(kwargs[k] for k in input_order)


def _default_device() -> torch.device:
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class BladeModel(PreTrainedModel):
    def __init__(self, opt_model: torch.jit.ScriptModule, info: ModelInfo) -> None:
        super().__init__(info.config)

        self.opt_model = opt_model
        self.info = info

    def forward(self, *args, **kwargs) -> Any:
        # TODO(litan.ls): merge kwargs to args according to tracable wrapper
        model_args = list(args)
        model_args.extend(_kwargs_to_args(self.info.input_order, **kwargs))
        outputs = self.opt_model(*model_args)
        return ModelOutput({k: v for k, v in zip(self.info.output_names, outputs)})


def create_model(task: Optional[str] = None, id_or_path: Optional[str] = None,
                 preprocessor: Optional[Preprocessor] = None,
                 **model_kwargs) -> Tuple[PreTrainedModel, ModelInfo]:
    info = load_model_info(task, id_or_path, preprocessor=preprocessor)
    _, model = infer_framework_load_model(info.id_or_path, info.config, model_classes=info.targeted_task,
                                          task=info.task, framework='pt', **model_kwargs)
    return model, info


def optimize(task: Optional[str] = None, id_or_path: Optional[str] = None,
             model: Optional[PreTrainedModel] = None,
             preprocessor: Optional[Preprocessor] = None,
             skip_compile: bool = False, amp: bool = False,
             device: Optional[torch.device] = None,
             model_kwargs: Dict[str, Any] = None) -> BladeModel:
    if device is None:
        device = _default_device()

    if model is None:
        model_kwargs = model_kwargs or {}
        model_kwargs['torchscript'] = True
        model, info = create_model(task=task, id_or_path=id_or_path,
                                   preprocessor=preprocessor, **model_kwargs)
    else:
        if task is None:
            raise ValueError('Must specify task when optimize loaded model.')
        if not model.config.torchscript:
            raise ValueError(
                'Only support optimize model with torchscript=True.')
        info = load_model_info(
            task=task, config=model.config, preprocessor=preprocessor)

    model.to(device)
    for k, v in info.dummy_inputs.items():
        info.dummy_inputs[k] = v.to(device)

    tracable = Tracable(model, [], info.default_kwargs, amp=amp)
    traced = torch.jit.trace(tracable.eval(), _kwargs_to_args(
        info.input_order, **info.dummy_inputs), strict=False)

    if skip_compile:
        opt_model = traced
    else:
        # TODO(litan.ls): support custom blade config
        torch_config = torch_blade.config.Config()
        torch_config.enable_mlir_amp = False  # disable mix-precision
        torch_config.customize_op_black_list = ['aten::arange']
        with torch.no_grad(), torch_config:
            opt_model = torch_blade.optimize(
                traced, model_inputs=_kwargs_to_args(info.input_order, **info.dummy_inputs))

    return BladeModel(opt_model, info)


def pipeline(
    task: Optional[str] = None,
    model: Optional[Union[str, PreTrainedModel]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    use_fast: bool = True,
    device: Optional[torch.device] = None,
    model_kwargs: Dict[str, Any] = None,
    skip_compile: bool = False,
    **kwargs,
) -> Pipeline:

    if device is None:
        device = _default_device()

    preprocessor = tokenizer or feature_extractor
    if isinstance(model, PreTrainedModel):
        blade_model = optimize(task=task, model=model, preprocessor=preprocessor,
                               device=device, model_kwargs=model_kwargs, skip_compile=skip_compile)
    elif model is None or isinstance(model, str):
        blade_model = optimize(task=task, id_or_path=model, preprocessor=preprocessor,
                               device=device, model_kwargs=model_kwargs, skip_compile=skip_compile)
    else:
        raise ValueError('model should be str|PretrainedModel|None')

    if preprocessor is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                blade_model.info.id_or_path, use_fast=use_fast, _from_pipeline=blade_model.info.task)
        except (OSError, KeyError):
            pass
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                blade_model.info.id_or_path, _from_pipeline=blade_model.info.task)
        except (OSError, KeyError):
            pass

    if tokenizer is None and feature_extractor is None:
        raise ValueError(
            f'Cannot create tokenizer or feature_extractor from {blade_model.info.id_or_path}')

    # TODO(litan.ls): fix pipeline check_model_type error msg.

    return hf_pipeline(
        task=blade_model.info.task,
        model=blade_model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device,
        model_kwargs=model_kwargs,
        **kwargs,
    )
