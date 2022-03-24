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

try:
    import tensorflow.compat.v1 as tf  # noqa: F401
except ImportError:
    import tensorflow as tf  # noqa: F401

__tf_major = int(tf.__version__.split(".")[0])
if __tf_major == 2:
    import tensorflow as tf2  # noqa: F401
elif __tf_major == 1:
    tf.disable_v2_behavior()
