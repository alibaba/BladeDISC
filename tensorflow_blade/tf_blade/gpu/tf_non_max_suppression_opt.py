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

'''
Author:
2.0 qiheng.fpf@alibaba-inc.com
1.0 Yifan Lu (evanlu.lyf@alibaba-inc.com)

This pass replaces TF native NonMaxSuppression ops with Blade custom ones

'''

import logging
from typing import ClassVar, Dict, Tuple

from tf_blade.util import tf_util
from tf_blade.util.tf_import_helper import tf


class TfNonMaxSuppressionOpt():
    nms_rep_map: ClassVar[Dict[str, str]] = {
        'NonMaxSuppression': 'BladeNonMaxSuppression',
        'NonMaxSuppressionV2': 'BladeNonMaxSuppressionV2',
        'NonMaxSuppressionV3': 'BladeNonMaxSuppressionV3',
        'NonMaxSuppressionV4': 'BladeNonMaxSuppressionV4',
    }

    def __init__(self) -> None:
        self.graph_def: tf.GraphDef

    def optimize_graph_def(self, graph_def: tf.GraphDef) -> Tuple[int, tf.GraphDef]:
        self.graph_def = graph_def
        count = tf_util.replace_node_ops_filter_dtype(
            self.graph_def, TfNonMaxSuppressionOpt.nms_rep_map, tf.float32
        )

        logging.info(f"{self.__class__.__name__}: total {count} NMS ops replaced with TfBlade's Fast NMS ones")

        if count > 0:
            return count, self.graph_def
        return 0, graph_def
