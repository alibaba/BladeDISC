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

import yaml
import os
import argparse

mini_backends = [
        "",
        "--backend blade",
        "--torchdynamo blade_optimize_dynamo"
        ]

full_backends = [
        "",
        "--backend blade",
        "--backend torchscript --no-ofi",
        "--torchdynamo eager",
        "--torchdynamo blade_optimize_dynamo",
        "--torchdynamo ts",
        "--torchdynamo inductor",
        "--torchdynamo ofi",
        "--torchdynamo ipex"
        ]
# tiny job:    one model,   eager/disc/dynamo-disc
# partial job: full models, eager/disc/dynamo-disc
# full job:    full models, all backends
def generate_yaml(path, job, models):
    assert(os.path.exists(path))
    yaml_file_name = "CPU_"+job+".yaml"
    yaml_file_path = os.path.join(path, "configs", yaml_file_name)
    if os.path.exists(yaml_file_path):
        os.remove(yaml_file_path)
    if job == "full":
        backends = full_backends
    else:
        backends = mini_backends
    dict_file = {
        'device' : ['cpu'],
        'test' : ['eval'],
        'models' : models,
        'precision' : ['fp32'],
        'args' : backends}
    with open(yaml_file_path, 'w') as file:
        yaml.dump(dict_file, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j", "--job", type=str,
        nargs='?', const="tiny", default="tiny"
    )
    parser.add_argument(
        "-p", "--path", type=str
    )
    parser.add_argument("-i", "--info", help="produce run info")
    args = parser.parse_args()
    current_file_dir = os.path.dirname(__file__)
    models_file = os.path.join(current_file_dir, args.job+"_models.txt")
    if args.job == "full":
        with open(models_file) as f:
            full_models = f.read().splitlines()
        generate_yaml(args.path, "full", full_models)
    elif args.job == "partial":
        with open(models_file) as f:
            partial_models = f.read().splitlines()
        generate_yaml(args.path, "partial", partial_models)
    else:
        with open(models_file) as f:
            partial_models = f.read().splitlines()
        generate_yaml(args.path, "tiny", tiny_models)

