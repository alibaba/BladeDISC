#!/bin/bash
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


# Returns the domain name of an URL.
function get_domain_from_url() {
    # cut extract the domain, domain is the third field split by '/'
    echo "$1" | cut -d '/' -f 3
}

# Find the fastest URL. The parameter consists of URLS separated by whitespace.
function find_fastest_url() {
    local speed=99999
    # shellcheck disable=SC2068
    for i in $@; do
        local domain
        domain=$(get_domain_from_url "$i")

        # c.f. https://stackoverflow.com/a/9634982/724872
        # redirect log output to stderr
        local cur_speed
        cur_speed=$(ping -c 4 -W 2 "$domain" | tail -1 \
                           | grep "/avg/" | awk '{print $4}'\
                           | cut -d '/' -f 2)
        cur_speed=${cur_speed:-99999}
        cur_speed=${cur_speed/.*/}

        # c.f. https://stackoverflow.com/a/31087503/724872
        if [[ $cur_speed -lt $speed ]]; then
            local best_domain="$i"
            speed="$cur_speed"
        fi
    done
    echo "$best_domain"
}

# Find fastest apt-get source, you can add mirrors in the 'apt_sources'
function find_fastest_apt_source() {
    # We need to specify \t as the terminate indicator character; otherwise, the
    # read command would return an non-zero exit code.
    read -r -d '\t' apt_sources <<EOM
http://mirrors.cloud.aliyuncs.com
http://mirrors.aliyun.com
http://archive.ubuntu.com
\t
EOM

    # Find the fastest APT source using ping.
    local fastest
    # shellcheck disable=SC2086
    fastest=$(find_fastest_url $apt_sources)/ubuntu/

    # The default Ubuntu version is 18.04, code named bionic.
    local codename
    source /etc/os-release
    codename=${UBUNTU_CODENAME-"bionic"}

    # Write APT source lists.
    cat <<EOF
deb $fastest $codename main restricted universe multiverse
deb $fastest $codename-security main restricted universe multiverse
deb $fastest $codename-updates main restricted universe multiverse
deb $fastest $codename-proposed main restricted universe multiverse
deb $fastest $codename-backports main restricted universe multiverse
deb-src $fastest $codename main restricted universe multiverse
deb-src $fastest $codename-security main restricted universe multiverse
deb-src $fastest $codename-updates main restricted universe multiverse
deb-src $fastest $codename-proposed main restricted universe multiverse
deb-src $fastest $codename-backports main restricted universe multiverse
EOF
}

set -xe

# Note(xiafei.qiuxf): Since we never install CUDA-related packages via apt-get, but
#                     actually from .tar.gz file on OSS, just remove cuda's apt repo to
#                     get free of its GPG key rotation issue.
rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

apt-get -qq update
apt-get install -y iputils-ping bc
if [[ "$ENABLE_FIND_FASTEST_APT_SOURCE" == "ON" ]]; then
    find_fastest_apt_source > /etc/apt/sources.list
fi
apt-get -qq update
