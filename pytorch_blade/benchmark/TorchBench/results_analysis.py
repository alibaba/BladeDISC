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
import math
import os
import pandas as pd

NOTICE_FILEDS = [
    "disc (compiled)",
    "disc (clusters)",
    "dynamo-disc (clusters)",
    "dynamo-disc (compiled)",
]
GITHUB_ISSUE_TEMPLATE = """
TorchBench CI has detected a performance signal.

Affected Tests:
{results}

detail data can be seen in {oss_dir}
created by TorchBench CI automatically
"""


def try_cast_to_float(to_cast: str):
    try:
        to_cast = float(to_cast)
        if math.isnan(to_cast):
            to_cast = "N/A"
    except:
        pass
    finally:
        return to_cast


def analyze_target(target, diff_percent, compare_fields):
    result = []
    baseline_file = f"{target}.csv"
    current_run_file = os.path.join(target, "summary.csv")
    assert os.path.exists(baseline_file) and os.path.exists(current_run_file)
    baseline_csv, current_run_csv = pd.read_csv(baseline_file), pd.read_csv(
        current_run_file
    )
    model_list = baseline_csv["Model"]
    # if model list changes
    if len(model_list) != len(current_run_csv["Model"]) or any(
        model_list != current_run_csv["Model"]
    ):
        result.append("model list changes, please update baseline files")
        return result

    for index in range(len(baseline_csv)):
        for field in compare_fields + NOTICE_FILEDS:
            model_name = model_list[index]
            baseline, current_run = (
                baseline_csv.iloc[index][field],
                current_run_csv.iloc[index][field],
            )
            baseline, current_run = try_cast_to_float(baseline), try_cast_to_float(
                current_run
            )
            # both cannot run
            if isinstance(baseline, str) and isinstance(current_run, str):
                continue
            elif isinstance(baseline, str) or isinstance(current_run, str):
                result.append(
                    f"\t- {model_name}[{field}] status changed, {baseline} -> {current_run}"
                )
            else:
                if field in NOTICE_FILEDS:
                    baseline, current_run = int(baseline), int(current_run)
                    if baseline != current_run:
                        result.append(
                            f"\t- {model_name}[{field}] {baseline} -> {current_run}"
                        )
                else:
                    diff = round((baseline - current_run) / baseline * 100, 4)
                    if abs(diff) > diff_percent:
                        sign = "+" if diff > 0 else ""
                        result.append(
                            f"\t- {model_name}[{field}] {baseline} -> {current_run}, {sign}{diff}%"
                        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--targets", nargs="+", default=[], help="Specify analysis target size"
    )
    parser.add_argument(
        "-p", "--percent", type=int, help="performance relative diff percent"
    )
    parser.add_argument(
        "-f", "--fields", nargs="+", default=[], help="Specify compare fields"
    )
    parser.add_argument("-i", "--info", help="produce run info")
    args = parser.parse_args()

    need_issue = False
    results = ""
    for target in args.targets:
        result = analyze_target(target, args.percent, args.fields)
        if result:
            need_issue = True
            results += f"\n- {target}:\n"
            results += "\n".join(result)
    print(need_issue)
    if need_issue:
        issue_str = GITHUB_ISSUE_TEMPLATE.format(results=results, oss_dir=args.info)
        print(issue_str)
        f = open("ISSUE.md", "w")
        f.write(issue_str)
        f.close()
