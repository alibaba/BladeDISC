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


date_str=$(date '+%Y%m%d')
rebase_branch=features/bladedisc_rebase_${date_str}
commit_msg="[BOT] bladedisc rebase ${date_str}"
export tf_commit_body=""

function rebase_tf() {
  set -ex
  base_branch=$(git rev-parse HEAD)
  git status
  git reset --hard HEAD
  git config remote.upstream.url >&- || git remote add upstream https://github.com/tensorflow/tensorflow.git
  git fetch upstream master
  export tf_commit_body="$(echo 'TensorFlow commits summary' && git log -n 30 --reverse --oneline ${base_branch}..upstream/master)"
  
  git checkout -B ${rebase_branch}
  git rebase upstream/master || git status && git rebase --abort && exit -1
  
  tf_remote_repo=https://bladedisc:${BOT_GITHUB_TOKEN}@github.com/pai-disc/tensorflow.git
  git push -uf ${tf_remote_repo} ${rebase_branch}
}

function create_pr() {
  set -ex
  git status
  git checkout -B ${rebase_branch}
  git add tf_community && git commit tf_community -m "${commit_msg}"
  remote_repo=https://alibaba:${GITHUB_TOKEN}@github.com/alibaba/BladeDISC.git
  git push -uf ${remote_repo} ${rebase_branch}

  gh pr create -B main -R ${remote_repo} -H ${rebase_branch} \
     --reviewer "fortianyou,qiuxiafei,wyzero,Yancey1989" \
     --title "${commit_msg}" \
     --body "${tf_commit_body}"
}

git config --global user.email "bladedisc-dev@list.alibaba-inc.com"
git config --global user.name "bladedisc-dev"
git submodule update --init --recursive
cd tf_community && rebase_tf && cd ..
create_pr
