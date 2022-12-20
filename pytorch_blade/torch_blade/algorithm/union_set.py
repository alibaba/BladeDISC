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

class UnionSet:
    def __init__(self, num_elems: int):
        self._num_elems = num_elems
        self._num_sets = num_elems
        self._group_id = dict([(g, g) for g in range(0, num_elems)])

    def same_group(self, x: int, y: int):
        pid_x = self.find(x)
        pid_y = self.find(y)

        if (pid_x == pid_y):
            return True
        return False

    def find(self, x: int):
        assert(x < self._num_elems)
        assert(x in self._group_id)

        pid = self._group_id[x]
        if (pid != self._group_id[pid]):
            pid = self.find(pid)

        # path compression
        self._group_id[x] = pid
        return pid

    def num_sets(self):
        return self._num_sets

    def union(self, x: int, y: int):
        pid_x = self.find(x)
        pid_y = self.find(y)

        if (pid_x == pid_y):
            return

        self._num_sets -= 1
        self._group_id[pid_y] = pid_x

    def get_groups(self):
        groups_dict = dict()
        for k in range(0, self._num_elems):
            pid = self.find(k)
            if pid not in groups_dict:
                groups_dict[pid] = list()
            groups_dict[pid].append(k)

        keys = sorted(groups_dict.keys())
        groups = list()
        for k in keys:
            assert(len(groups_dict[k]) > 0)
            groups.append(groups_dict[k])

        assert(self._num_sets == len(groups))
        return groups
