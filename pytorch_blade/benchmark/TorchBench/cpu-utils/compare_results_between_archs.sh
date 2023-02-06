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

# !/bin/bash
if [ -f $HOME/.cache/proxy_config ]; then
  source $HOME/.cache/proxy_config
fi
ARCH="X86-intel" ## X86-intel, X86-amd, AArch64-g6r, AArch64-yitian
JOB="partial"        ## tiny, partial, full
DATE=""
while getopts ":a:j:d:h" optname
do
    case "$optname" in
        "a")
            ARCH=$OPTARG
            ;;
        "j")
            JOB=$OPTARG
            ;;
        "d")
            DATE=$OPTARG
            ;;
	"h")
	    echo "parse_cpu_results.sh -a X86-intel;AArch64-g6r -j tiny -d 20230207;20230207"
	    echo "-a: X86-intel, X86-amd, AArch64-g6r, AArch64-yitian"
	    echo "-j: tiny, partial, full"
	    echo "-d: yyyymmdd-num"
	    ;;
        "?")
            echo "Unknown option $OPTARG"
            ;;
         *)
            echo "Unknown error while processing options"
            ;;
    esac
done

if [[ -z $DATE ]]; then
    echo "require date specification!!!"
    exit 1
fi

## parse architecture and its corresponding date string
ARCHlist=($(echo $ARCH | tr ";" "\n"))
DATElist=($(echo $DATE | tr ";" "\n"))
if [[ ${#ARCHlist[@]} != ${#DATElist[@]} ]]; then
  echo "ARCH list or DATE list incompatiable!!"
  exit 1
else
  echo ${#ARCHlist[@]} ${#DATElist[@]}
fi

oss_link=https://bladedisc-ci.oss-cn-hongkong.aliyuncs.com
OSSUTIL=ossutil
GH=gh
if [[ $HARDWARE == AArch64* ]]; then
    OSSUTIL=ossutil-arm64
    GH=gh_arm64
fi
for (( i=0;i<${#ARCHlist[@]};i++)) do
    echo ${ARCHlist[i]}
    echo ${DATElist[i]}
    CUR_ARCH=${ARCHlist[i]}
    CUR_DATE=${DATElist[i]}
    CUR_FILE=${CUR_ARCH}.${JOB}.${CUR_DATE}
    ## oss://bladedisc-ci/TorchBench/cpu/AArch64-g6r/tiny/20230207-08/
    oss_dir=${oss_link}/TorchBench/cpu/${CUR_ARCH}/${JOB}/${CUR_DATE}
    echo $oss_dir
    echo ${CUR_FILE}.tar.gz
    echo curl ${oss_dir}/${CUR_FILE}.tar.gz -o ${CUR_FILE}.tar.gz
    curl ${oss_dir}/${CUR_FILE}.tar.gz -o ${CUR_FILE}.tar.gz
    tar -xf ${CUR_FILE}.tar.gz
done

## using python to parse csv files
analyze_target=("speedup" "latency")
for ((i=0;i<${#analyze_target[@]};++i)) do
  python3 parse_cpu_results.py --a $ARCH --j $JOB --d $DATE -t ${analyze_target[i]}
  archs=$(IFS=, - echo "${ARCHlist[*]}")
  dates=$(IFS=, - echo "${DATElist[*]}")
  summary_name=${archs}.${dates}.${JOB}.${analyze_target[i]}.csv
  echo "$summary_name"
done
