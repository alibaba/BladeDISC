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

import argparse
from inspect import signature
from typing import Dict, List
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow_hub import resolve
import yaml

# optimize_methods: List[str] = [None, "disc", "xla", "tf_trt", "tf_blade", "tvm"]
optimize_methods: List[str] = [
    "xla",
    "disc",
]


class TFBenchmark:
    def __init__(self, config_file, models_dir, result_file) -> None:
        self.config_file = config_file
        self.model_info: Dict[str, str]
        self.model_list: List[str]
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        self.models_dir = models_dir
        self.load_config(config_file)
        self.result_file = result_file
        self.prepare_results()

    def load_config(self, config_file: str):
        with open(config_file) as f:
            self.model_info = yaml.safe_load(f)
        self.model_list = list(self.model_info.keys())
        print(self.model_list)

    def download_model(self, model_name: str) -> str:
        model_path = os.path.join(self.models_dir, model_name)

        if not os.path.exists(model_path):
            model_url = self.model_info[model_name]["url"]
            tf_model_path = resolve(model_url)
            # model_path.mkdir(exist_ok=True, parents=True)
            # avoid subprocess crush
            gpus = tf2.config.list_physical_devices("GPU")
            tf2.config.experimental.set_memory_growth(gpus[0], True)
            # download model
            model = hub.load(model_url)
            del model
            shutil.copytree(tf_model_path, model_path)

        return model_path

    def prepare_results(self):
        if os.path.exists(self.result_file):
            os.remove(self.result_file)
        headers = "Model,tf(Latency)"
        for optimize_method in optimize_methods:
            headers += f",{optimize_method}(Latency),{optimize_method}(speedup)"
        headers += "\n"
        with open(self.result_file, "w") as rf:
            rf.writelines(headers)

    def prepare_model(self, model_name: str, amp: bool = False) -> None:
        # download model
        self.download_model(model_name)

    def run(self) -> None:
        for model_name in self.model_list:
            print("=" * 80)
            print(f"Begin benchmark model {model_name} now!")
            self.prepare_model(model_name)
            subprocess.check_call(
                [
                    sys.executable,
                    "TFModel.py",
                    "-m",
                    model_name,
                    "-c",
                    self.config_file,
                    "-p",
                    self.models_dir,
                    "-r",
                    self.result_file,
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", "-c", help="model config yaml file")
    parser.add_argument("--model-path", "-p", help="benchmark model path")
    parser.add_argument("--result-file", "-r", help="result file path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    bench = TFBenchmark(args.config_file, args.model_path, args.result_file)
    bench.run()
