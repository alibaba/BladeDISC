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

import os 
import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

SUMMARY_CSVS=[]
PATH=""
TARGETS=[]
RESULT_SUMMARY_DICT={}

def check_correctness(row):
    if row.empty:
        return ''
    elif row.eq(True).all() or row.isnull().all():
        return ''
    elif row.eq(False).all():
        return 'All Wrong'
    else:
        find_false=False
        find_true=False
        for index, val in row.items():
            if np.isnan(val):
                continue
            elif val == False:
                find_false=True
            else:
                find_true=True
        if find_false and find_true:
            return 'Some Wrong'
        elif find_false and not find_true:
            return ''
        elif find_true and not find_false:
            return ''

def search_summary_csv(path):
    if not os.path.isdir(path):
        return
    for filename in os.listdir(path):
        fp = os.path.join(path, filename)
        if os.path.isfile(fp) and filename == "summary.csv":
            SUMMARY_CSVS.append(fp)
        elif os.path.isdir(fp):
            search_summary_csv(fp)
        elif os.path.isfile(fp):
            continue
        else:
            raise Exception("error when search summary csv")

def plot_speedup_result(df, core_binding, backends, filename):
    models = df['Model']
    backend_speedups = df[backends].to_dict('list')
    x_labels = np.arange(len(models))  # the label locations
    width = 0.8 / len(backends)  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(constrained_layout=True,figsize=(100,60))
    disc_speedup_mean=0
    disc_speedup_mid=0
    dynamo_disc_speedup_mean=0
    dynamo_disc_speedup_mid=0
    speedup_max=0
    speedup_min=0
    for backend, speedup in backend_speedups.items():
        #offset = width * multiplier
        #rects = ax.bar(x + offset, measurement, width, label=attribute)
        offset = width * multiplier
        rects = ax.bar(x_labels + offset, speedup, width, label=backend)
        ax.bar_label(rects, padding=3)
        speedup_max = max(speedup_max, np.nanmax(speedup))
        speedup_min = min(speedup_min, np.nanmin(speedup))
        correctness_name = [name for name in list(df.columns) if name.startswith(backend.split(' ')[0]) \
                            and name.find("correctness") != -1]
        assert len(correctness_name) == 1
        correctness = df[correctness_name[0]].tolist()
        for x, y, c in zip(x_labels, speedup, correctness):
            if not c:
                plt.text(x+offset, 0, "False", rotation_mode='anchor', \
                         ha='center', bbox=dict(facecolor='red', alpha=.8), rotation=270)
        multiplier += 1
        if backend.find("disc") != -1:
            if backend.find("dynamo") == -1:
                disc_speedup_mean=np.nanmean(speedup)
                disc_speedup_mid=np.nanmedian(speedup)
            else:
                dynamo_disc_speedup_mean=np.nanmean(speedup)
                dynamo_disc_speedup_mid=np.nanmedian(speedup)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('speedup (%)', size=50)
    ax.set_title(core_binding + " speedup", size=70)
    ax.set_xticks(x_labels+width*max(1, len(backends)/2-1), models)
    plt.xticks(size=60, rotation=270)
    plt.yticks(size=100)
    plt.axhline(y=disc_speedup_mean, color='red', linestyle='--', linewidth=3, label='Disc Speedup Mean')
    plt.text(0, disc_speedup_mean, str(disc_speedup_mean), ha='center', fontsize=10)
    plt.axhline(y=disc_speedup_mid,  color='red', linewidth=3, label='Disc Speedup Mid')
    plt.axhline(y=dynamo_disc_speedup_mean, color='blue', linestyle='--', linewidth=3, label='Dynamo Disc Speedup Mean')
    plt.text(0, dynamo_disc_speedup_mean, str(disc_speedup_mean), ha='center', fontsize=10)
    plt.axhline(y=dynamo_disc_speedup_mid,  color='blue', linewidth=3, label='Dynamo Disc Speedup Mid')
    ax.legend(loc='upper left', ncols=len(backends), prop={'size': 60})
    ax.set_ylim(speedup_min*1.2, speedup_max*1.2)
    #plt.grid(axis='y')
    plt.grid()
    plt.savefig(PATH + "/" + filename)
    plt.close(fig)

def plot_disc_speedup_result(df, core_binding):
    backends = [name for name in list(df.columns) if name.find("speedup") != -1 and name.find("disc") != -1]
    plot_speedup_result(df, core_binding, backends, core_binding+"_DISC.png")

def plot_all_speedup_result(df, core_binding):
    backends = [name for name in list(df.columns) if name.find("speedup") != -1]
    plot_speedup_result(df, core_binding, backends, core_binding+"_ALL.png")

def plot_disc_latency_result(df):
    core_bindings=df['core binding'].unique()
    print(core_bindings)
    models = df['Model'].tolist()
    backends = [column for column in list(df.columns) if column.find("latency") != -1]
    for model in models:
        for backend in backends:
            latency = pd.to_numeric(df.loc[(df['core binding'].isin(core_bindings)) & \
                          (df['Model']==model)][backend], downcast='signed', errors ='coerce').tolist()
            if np.isnan(latency).any():
                continue
            fig, ax = plt.subplots(figsize=(100, 50))
            ax.bar(core_bindings, latency)
            ax.set_ylabel('Latency (ms)')
            ax.set_title(model+" "+backend + ' latency with different core bindings', size=70)
            plt.xticks(size=50, rotation=270)
            plt.yticks(size=60)
            ax.legend(loc='upper left', ncols=len(core_bindings), prop={'size':50})
            ax.set_ylim(0, np.max(latency)*1.2)
            plt.grid()
            plt.savefig(PATH+"/"+model+"_"+backend.split(' ')[0]+"_latency_with_core_bindings.png")
            plt.close(fig)

def concate_csv():
    frames=[]
    for summary_csv in SUMMARY_CSVS:
        core_binding = re.search(r"threads_(.*?)_cores_([0-9]*-?[0-9]*)", summary_csv)
        threads = core_binding.group(1)
        cores = core_binding.group(2)
        print("threads=", threads, "cores", cores)
        df = pd.read_csv(summary_csv)
        target_frames=[]
        for target in TARGETS:
            column_names = [name for name in list(df.columns) if name.find(target) != -1]
            if target == "speedup":
                target_df = (df[column_names].copy() - 1) * 100
                target_df = pd.concat([df["Model"], target_df], axis=1)
                correctness = [name for name in list(df.columns) if name.find("correctness") != -1]
                plot_disc_speedup_result(pd.concat([target_df]+[df[correctness].copy()], axis=1), threads+"_"+cores)
                plot_all_speedup_result(pd.concat([target_df]+[df[correctness].copy()], axis=1),  threads+"_"+cores)
            else:
                column_names = ["Model"] + column_names
                target_df = df[column_names].copy()
            target_frames.append(target_df)
        core_binding_column = [threads + ":" + cores] * len(target_frames[0])
        new_df = pd.concat(target_frames, axis=1)
        new_df.insert(loc=1, column="core binding", value=core_binding_column)
        frames.append(new_df)
    df = pd.concat(frames)
    df = df.reset_index(drop=True)
    return df

def concate_csv_according_to_threads():
    df = pd.read_csv(SUMMARY_CSVS[0])
    backends = [name.split(' ')[0] for name in list(df.columns) if name.find("latency") != -1]
    models = df['Model']
    for backend in ["disc", "dynamo-disc"]:
        RESULT_SUMMARY_DICT[backend] = {}
        for csv in SUMMARY_CSVS:
            df = pd.read_csv(csv)
            core_binding = re.search(r"threads_(.*?)_cores_([0-9]*-?[0-9]*)", csv)
            threads = core_binding.group(1)
            cores = core_binding.group(2)
            RESULT_SUMMARY_DICT[backend][threads+":"+cores] = {}
            RESULT_SUMMARY_DICT[backend][threads+":"+cores]["0.05-0.2"] = []
            RESULT_SUMMARY_DICT[backend][threads+":"+cores]["0.2-0.5"] = []
            RESULT_SUMMARY_DICT[backend][threads+":"+cores]["0.5-1"] = []
            RESULT_SUMMARY_DICT[backend][threads+":"+cores]["gt-1"] = []
            RESULT_SUMMARY_DICT[backend][threads+":"+cores]["lt-negative0.05"] = []
            RESULT_SUMMARY_DICT[backend][threads+":"+cores]["+-0.05"] = []
            RESULT_SUMMARY_DICT[backend][threads+":"+cores]["WrongResult"] = []
            speedup = []
            correctness = []
            models = df['Model'].tolist()
            eager_latency = df['eager (latency)'].tolist()
            disc_latency = df['disc (latency)'].tolist()
            for column in list(df.columns):
                if column.startswith(backend):
                    if column.find("speedup") != -1:
                        speedup = df[column].tolist()
                    if column.find("correctness") != -1:
                        correctness = df[column].tolist()
            for s, c, m in zip(speedup, correctness, models):
                if c:
                    if (s-1) > 0.05 and (s-1) <= 0.2:
                        RESULT_SUMMARY_DICT[backend][threads+":"+cores]["0.05-0.2"].append(m)
                    elif (s-1) > 0.2 and (s-1) <= 0.5:
                        RESULT_SUMMARY_DICT[backend][threads+":"+cores]["0.2-0.5"].append(m)
                    elif (s-1) > 0.5 and (s-1) <= 1:
                        RESULT_SUMMARY_DICT[backend][threads+":"+cores]["0.5-1"].append(m)
                    elif (s-1) > 1:
                        RESULT_SUMMARY_DICT[backend][threads+":"+cores]["gt-1"].append(m)
                    elif (s-1) < -0.05:
                        RESULT_SUMMARY_DICT[backend][threads+":"+cores]["lt-negative0.05"].append(m)
                    elif not np.isnan(s):
                        RESULT_SUMMARY_DICT[backend][threads+":"+cores]["+-0.05"].append(m)
                elif np.isnan(c):
                    continue
                elif not c:
                    RESULT_SUMMARY_DICT[backend][threads+":"+cores]["WrongResult"].append(m)
            
    for backend in backends:
        frames = []
        core_bindings=[]
        for csv in SUMMARY_CSVS:
            df = pd.read_csv(csv)
            for column in list(df.columns):
                if column.startswith(backend) and column.find("latency") != -1:
                    core_binding = re.search(r"threads_(.*?)_cores_([0-9]*-?[0-9]*)", csv)
                    core_bindings.append(core_binding.group(1)+":"+core_binding.group(2))
                    frames.append(df[column])
        correctness_frames=[]
        for csv in SUMMARY_CSVS:
            df = pd.read_csv(csv)
            for column in list(df.columns):
                if column.startswith(backend) and column.find("correctness") != -1:
                    core_binding = re.search(r"threads_(.*?)_cores_([0-9]*-?[0-9]*)", csv)
                    core_bindings.append(core_binding.group(1)+":"+core_binding.group(2))
                    correctness_frames.append(df[column])
        if len(correctness_frames) != 0:
            correctness_df = pd.concat(correctness_frames, axis=1)
            correctness_df['not matching'] = correctness_df.apply(lambda row : check_correctness(row), axis=1)
            for disc in ["disc", "dynamo-disc"]:
                if backend.startswith(disc):
                    RESULT_SUMMARY_DICT[disc]["All Wrong"] = []
                    RESULT_SUMMARY_DICT[disc]["Some Wrong"] = []
                    for m1, m2 in zip(models, correctness_df['not matching'].tolist()):
                        if m2 == "All Wrong":
                            RESULT_SUMMARY_DICT[disc]["All Wrong"].append(m1)
                        elif m2 == "Some Wrong":
                            RESULT_SUMMARY_DICT[disc]["Some Wrong"].append(m1)
            frames.append(correctness_df)
        total_df = pd.concat(frames, axis=1)
        if len(list(total_df.columns)) > len(SUMMARY_CSVS):
            header=[core_bindings+[''], total_df.columns]
            total_df.columns=header
        total_df.insert(loc=0, column="Model", value=models)
        total_df.to_csv(PATH+"/"+backend+"_multi_threads_result.csv")

def show_result_summary():
    original_stdout = sys.stdout
    with open(PATH+'/result_summary.txt', 'w') as f:
        sys.stdout = f
        for backend, _ in RESULT_SUMMARY_DICT.items():
            for core_binding, _ in RESULT_SUMMARY_DICT[backend].items():
                if core_binding in ["All Wrong", "Some Wrong"]:
                    continue
                print(backend, "with core binding: ", core_binding) 
                print("    [5%,   20%): ", RESULT_SUMMARY_DICT[backend][core_binding]["0.05-0.2"])
                print("    [20%,  50%): ", RESULT_SUMMARY_DICT[backend][core_binding]["0.2-0.5"])
                print("    [50%, 100%): ", RESULT_SUMMARY_DICT[backend][core_binding]["0.5-1"])
                print("    [100%,   +): ", RESULT_SUMMARY_DICT[backend][core_binding]["gt-1"])
                print("    [-,    -5%): ", RESULT_SUMMARY_DICT[backend][core_binding]["lt-negative0.05"])
                print("    [-5%,  +5%): ", RESULT_SUMMARY_DICT[backend][core_binding]["+-0.05"])
            print("  All  Wrong: ", len(RESULT_SUMMARY_DICT[backend]["All Wrong"]), RESULT_SUMMARY_DICT[backend]["All Wrong"])
            print("  Some Wrong: ", len(RESULT_SUMMARY_DICT[backend]["Some Wrong"]), RESULT_SUMMARY_DICT[backend]["Some Wrong"])
    sys.stdout = original_stdout

def parse_csv():
    df = concate_csv()
    df.to_csv(PATH + "/Total_SLC.csv")
    ## TODO: plot latency with different core bindings
    #columns = [column for column in list(df.columns) if column.find("latency") != -1]
    #plot_disc_latency_result(pd.concat([df.iloc[:,0:2], df[columns]], axis=1))
    concate_csv_according_to_threads()
    show_result_summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str
    )
    parser.add_argument(
        "-t", "--targets", type=str, help="Specify compare fields",
        nargs='?', const="speedup;latency;correctness", default="speedup;latency;correctness"
    )
    parser.add_argument("-i", "--info", help="produce run info")
    args = parser.parse_args()
    PATH = args.path
    search_summary_csv(PATH)
    TARGETS = args.targets.split(";")
    if len(SUMMARY_CSVS) != 0:
        SUMMARY_CSVS.sort(key=lambda x: int(re.search(r'threads_(.*?)_cores', x).group(1)))
        print(SUMMARY_CSVS)
        parse_csv()
