# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


tao_execute_cmd(
    DESC "get tvm folder"
    PWD "."
    COMMAND ${PYTHON} -c "from __future__ import print_function\nimport os\nimport tvm\nprint(os.path.dirname(tvm.__file__), end='')"
    STD_OUT OUTPUT
  )
set(TVM_PKG_DIR ${OUTPUT})

message(STATUS "Working on tvm ${TVM_PKG_DIR}")

set(DLPACK_PATH "${TVM_PKG_DIR}/3rdparty/dlpack/include")
set(DMLC_PATH "${TVM_PKG_DIR}/3rdparty/dmlc-core/include")

set(TVM_LINK_DIR ${TVM_PKG_DIR})
list(APPEND TVM_INCLUDE_DIRS
    ${TVM_PKG_DIR}/include
    ${DLPACK_PATH}
    ${DMLC_PATH})
