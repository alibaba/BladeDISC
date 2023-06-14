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

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR=''

def STD(row):
    row = pd.to_numeric(row, errors='coerce')
    mean = row.mean()
    std=np.std((row-mean)*100/mean).round(decimals=2)
    return std

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

def latency_ratio_to_intel(row):
    row = pd.to_numeric(row, errors='coerce')
    base_latency = 0
    for index, val in row.items():
        if index[0].find('intel') != -1 and not np.isnan(val):
            base_latency = val
    for index, val in row.items():
        if index[0].find('intel') != -1:
            continue
        if base_latency != 0 and not np.isnan(val):
            replace_value = round(val/base_latency, 2)
            row[index]=replace_value
        elif base_latency != 0 and np.isnan(val):
            None
        elif base_latency == 0 and not np.isnan(val):
            row[index]=1
    return row

def plot_std_result(df, arch, column):
    core_bindings = df['core binding'].unique()
    std_dict = {}
    models=[]
    backend=column.split(' ')[0]
    std_max = 0
    for core_binding in core_bindings:
        core_binding_df = df.loc[df['core binding']==core_binding]
        std_list = core_binding_df['std'].tolist()
        models=core_binding_df['Model']
        std_dict[core_binding] = std_list
        std_max = max(std_max, np.nanmax(std_list))
    print("plot " +arch+" " + backend + " latency std")
    x = np.arange(len(models))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(constrained_layout=True,figsize=(len(core_bindings)*15, 30))
    for core_binding, std in std_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, std, width, label=core_binding)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('STD (%)', size=50)
    ax.set_title(arch+" " + backend + " latency std (%)", size=70)
    ax.set_xticks(x_labels+width*max(1, len(backends)/2-1), models)
    plt.xticks(size=50, rotation=270)
    plt.yticks(size=60)
    ax.legend(loc='upper left', ncols=len(core_bindings), prop={'size':50})
    ax.set_ylim(0, std_max * 1.2)
    plt.grid()
    plt.savefig(OUTPUT_DIR+"/latency_std_"+arch+"_"+backend+".png")
    plt.close(fig)

def plot_latency_ratio_across_archs(df, archs):
    core_bindings = df['core binding'].unique()
    ## multi archs, multi models, one backend, one core_binding each image
    for core_binding in core_bindings:
        latency_ratio = {} ## arch : [latency]
        correctness_dict = {} ## arch : [correcntess]
        models = []
        backend=''
        ratio_max = 0
        for arch in archs:
            if arch.find('intel') != -1:
                continue
            core_binding_df = df.loc[df['core binding']==core_binding]
            models = core_binding_df['Model']
            for column in df.columns:
                if not type(column) is tuple:
                    continue
                if column[0].find(arch+"_") != -1 and column[1].find("latency") != -1:
                    latency_ratio[column[0]] = core_binding_df[column].tolist()
                    ratio_max = np.nanmax(core_binding_df[column].tolist() + [ratio_max])
                    backend=column[1].split(' ')[0]
                if column[0].find(arch+"_") != -1 and column[1].find("correctness") != -1:
                    correctness_dict[column[0]] = core_binding_df[column].tolist()
        print("plot " + " " + backend  + " " + core_binding + " latency ration (Intel Base)")
        if np.isnan(ratio_max):
            continue
        x_labels = np.arange(len(models))  # the label locations
        width = 0.9/len(archs)  # the width of the bars
        multiplier = 0

        #fig, ax = plt.subplots(constrained_layout=True,figsize=(len(archs)*30, 60))
        fig, ax = plt.subplots()
        fig.set_figheight(60)
        fig.set_figwidth(len(archs)*30)
        for arch, ratio in latency_ratio.items():
            offset = width * multiplier
            rects = ax.bar(x_labels + offset, ratio, width, label=arch)
            ax.bar_label(rects, padding=1)
            ## TODO: add correctness
            ##   core_binding, backend, arch
            if arch in correctness_dict:
                correctness = correctness_dict[arch]
                for x, y, c in zip(x_labels, ratio, correctness):
                    if not np.isnan(c) and (not c or c == 0.0 or c == "False"):
                        plt.text(x+offset, 0, "Flase", rotation=270, ha='center', bbox=dict(facecolor='red', alpha =.8))
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Ratio', size=50)
        ax.set_title(backend+" "+core_binding+" latency ratio (intel base)", size=70)

        ax.set_xticks(x_labels+width*max(len(archs)*0.5-1, 1), models)
        plt.xticks(size=50, rotation=270)
        plt.yticks(size=60)
        ax.legend(loc='upper left', ncols=len(archs), prop={'size':50})
        ax.set_ylim(0, ratio_max*1.2)
        plt.grid()
        plt.savefig(OUTPUT_DIR+"/latency_ratio_"+backend+"_"+core_binding.replace(':', '_')+".png")
        plt.close(fig)

def plot_latency_ratio_across_core_bindings(df, archs):
    core_bindings = df['core binding'].unique()
    ## multi core_binding, multi models, one backend, one arch each image
    for arch in archs:
        if arch.find("intel") != -1:
            continue
        latency_ratio = {}
        correctness_dict = {}
        models=[]
        backend=''
        ratio_max=0
        for core_binding in core_bindings:
            core_binding_df = df.loc[df['core binding']==core_binding]
            models=core_binding_df['Model']
            for column in df.columns:
                if not type(column) is tuple:
                    continue
                if column[0].find(arch+"_") != -1 and column[1].find("latency") != -1:
                    latency_ratio[core_binding] = core_binding_df[column].tolist()
                    ratio_max=np.nanmax(core_binding_df[column].tolist() + [ratio_max])
                    backend=column[1].split(' ')[0]
                if column[0].find(arch+"_") != -1 and column[1].find("correctness") != -1:
                    correctness_dict[core_binding] = core_binding_df[column].tolist()
        print("plot " + arch + " " + backend+" latency ratio (intel base)")
        x_labels = np.arange(len(models))  # the label locations
        width = 0.9 / len(core_bindings)  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots()
        fig.set_figheight(60)
        fig.set_figwidth(len(archs)*30)

        for core_binding, ratio in latency_ratio.items():
            offset = width * multiplier
            rects = ax.bar(x_labels + offset, ratio, width, label=core_binding)
            ax.bar_label(rects, padding=1)
            multiplier += 1
            if core_binding in correctness_dict:
                correctness = correctness_dict[core_binding]
                for x, y, c in zip(x_labels, ratio, correctness):
                    if not np.isnan(c) and (c == 0.0 or not c or c=="False"):
                        plt.text(x+offset, 0, "False", rotation=270, ha='center', bbox=dict(facecolor='red', alpha=.8))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Ratio', size=50)
        ax.set_title(arch + " " +backend+" " +" latency ratio (intel base)", size=70)
        ax.set_xticks(x_labels + width*(max(len(core_bindings)*0.5 - 1, 1)), models)
        plt.xticks(size=50, rotation=270)
        plt.yticks(size=60)
        ax.legend(loc='upper left', ncols=len(core_bindings), prop={'size':50})
        ax.set_ylim(0, ratio_max*1.2)
        plt.grid()
        plt.savefig(OUTPUT_DIR+"/latency_ratio_"+arch+"_"+backend+".png")
        plt.close(fig)

def compare_target_between_archs(archs, dates, job, targets):
    data_frames = []
    for i in range(len(archs)):
        arch = archs[i]
        date = dates[i]
        dir_path = os.path.join(os.getcwd(), arch+"."+job+"."+date)
        if not os.path.exists(dir_path):
            continue
        total_slc_fp = os.path.join(dir_path, 'Total_SLC.csv')
        if os.path.exists(total_slc_fp):
            total_slc_csv = pd.read_csv(total_slc_fp)
            ## for intel 1 or 2 thread, select the min latency
            if arch.find("intel") != -1 and len(archs)==len(set(archs)) and len(archs) > 1:
                latency_columns = [column for column in list(total_slc_csv.columns) if column.find("latency") != -1]
                one_thread_results=total_slc_csv.loc[total_slc_csv['core binding']=='1:0'][latency_columns].copy()
                two_threads_results=total_slc_csv.loc[total_slc_csv['core binding']=='2:0-1'][latency_columns].copy()
                for column in latency_columns:
                    s1 = pd.to_numeric(one_thread_results[column], errors='coerce').to_list()
                    s2 = pd.to_numeric(two_threads_results[column], errors='coerce').tolist()
                    s3 = []
                    for j in range(len(s1)):
                        if s1[j] < s2[j]:
                            s3.append(s1[j])
                        else:
                            s3.append(s2[j])
                    total_slc_csv.loc[total_slc_csv['core binding']=='1:0', column] = s3
                    total_slc_csv.loc[total_slc_csv['core binding']=='2:0-1', column] = s3
                    #print(total_slc_csv.loc[total_slc_csv['core binding']=='1:0', column])
                    #print(total_slc_csv.loc[total_slc_csv['core binding']=='2:0-1', column])
            header=[[arch+"_"+str(i)] * len(total_slc_csv.columns), total_slc_csv.columns]
            total_slc_csv.columns = header
            data_frames.append(total_slc_csv)

    if len(data_frames) == 0:
        return
    total_slc_csv_accross_archs = pd.concat(data_frames, axis=1)
    total_target_frames = []
    models = total_slc_csv_accross_archs[archs[0]+"_0"]["Model"]
    core_binding=total_slc_csv_accross_archs[archs[0]+"_0"]["core binding"]
    for target in targets:
        target_frames = []
        columns = [name for name in list(total_slc_csv_accross_archs[archs[0]+"_0"].columns) if name.find(target) != -1]
        correctness_columns = ['']+[name for name in list(total_slc_csv_accross_archs[archs[0]+"_0"].columns) if name.find("correctness") != -1]
        ## iterate all backends: eager, ofi, disc, dynamo...
        for column, correctness in zip(columns, correctness_columns):
            target_frame = total_slc_csv_accross_archs.loc(axis=1)[:,column]
            if target == "latency" and len(set(archs))==1 and len(archs)>1:
                ## same arch, same core binding, same backend: compute latency std
                target_frame['std'] = target_frame.apply(lambda row : STD(row), axis=1)
                plot_std_result(pd.concat([models, core_binding, target_frame['std']], axis=1), archs[0], column)
            elif target == 'latency' and len(archs)==len(set(archs)) and len(archs) > 1:
                ## different arch, same core binding, same backend: compute latency ratio
                target_frame = target_frame.apply(lambda row : latency_ratio_to_intel(row), axis=1)
                new_target_frame = target_frame
                if len(correctness) != 0:
                    new_target_frame = pd.concat([target_frame, total_slc_csv_accross_archs.loc(axis=1)[:,correctness]], axis=1)
                plot_latency_ratio_across_archs(pd.concat([models, core_binding, new_target_frame], axis=1), archs)
                plot_latency_ratio_across_core_bindings(pd.concat([models, core_binding, new_target_frame], axis=1), archs)
            elif target == 'correctness' and len(archs)>1:
                #target_frame['matching'] = target_frame.eq(target_frame.iloc[:, 0], axis=0).all(1).astype(int)
                target_frame['not matching'] = target_frame.apply(lambda row : check_correctness(row), axis=1)
            elif target == 'speedup':
                ## TODO: compare eager and disc and dynamo disc
                ## diff arch, same core binding ,same backend: speedup
                None
            else:
                None
            target_frames.append(target_frame)
        #print(pd.concat(target_frames, axis=1))
        total_target_frames.append(pd.concat([models, core_binding] + target_frames, axis=1))
    df = pd.concat(total_target_frames, axis=1)
    #df.insert(loc=0, column="Model", value=total_slc_csv_accross_archs[archs[0]+"_0"]["Model"])
    #df.insert(loc=0, column="Model", value=models)
    #df.insert(loc=1, column="core binding", value=core_binding)
    if len(set(archs))==1 and len(archs)>1:
        df.to_csv(OUTPUT_DIR+"/"+archs[0]+"_multi_times_result.csv")
    elif len(archs)==len(set(archs)) and len(archs) > 1:
        df.to_csv(OUTPUT_DIR+"/"+"_".join(archs)+"_result.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--archs", type=str, help="Specify architecture", nargs='?'
    )
    parser.add_argument(
        "-d", "--dates", type=str, help="Specify date", nargs='?'
    )
    parser.add_argument(
        "-j", "--job", type=str, nargs='?'
    )
    parser.add_argument(
        "-o", "--output_path", type=str, nargs='?'
    )
    parser.add_argument(
        "-t", "--targets", type=str, help="Specify compare fields",
        nargs='?', const="latency;correctness", default="latency;correctness"
    )
    parser.add_argument("-i", "--info", help="produce run info")
    args = parser.parse_args()
    archs = args.archs.split(";")
    print(archs)
    assert len(archs)==len(set(archs)) or len(set(archs))==1
    dates = args.dates.split(";")
    targets = args.targets.split(";")
    job = args.job
    OUTPUT_DIR=args.output_path
    if len(archs) != len(dates):
        raise Exception("arch nums should equal to date nums")
    compare_target_between_archs(archs, dates, job, targets)
