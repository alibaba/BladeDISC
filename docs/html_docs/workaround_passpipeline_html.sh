# Copyright 2021 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# In order to show the intermediate IRs for pass_pipelien.md in
# the website: 
# 
# step 1, setup a environment with CUDA 10.0, jupyter notebook and jupytext
# step 2, run the pass_pipeline.md with Jupyter Notebook, and save it to
#         pass_pipeline.ipynb
# step 3, jupyter nbconvert --to html pass_pipeline.ipynb
# step 4, ./workaround_passpipeline_html.sh

# TODO: We should revisit if this is necessary in future. If not, ReadtheDocs
# is a better solution for maintaining a document website.

sed -i 's/<div\ class=\"jp-RenderedText\ jp-OutputArea-output\"\ data-mime-type=\"text\/plain\">/<div\ class=\"jp-RenderedText\ jp-OutputArea-output\"\ data-mime-type=\"text\/plain\"\ style=\"overflow-y:\ scroll;\ height:400px;\">/g' pass_pipeline.html
sed -i 's/pics/..\/..\/_images/g' pass_pipeline.html
cp pass_pipeline.html ../html_docs/build/html/docs/html/pass_pipeline.html
