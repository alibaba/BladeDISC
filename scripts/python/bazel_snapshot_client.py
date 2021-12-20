#!/usr/bin/env python3

# required packages in update mode
# pip3 install oss2

import argparse
import logging
import re
import os

from tao_common import running_on_ci


env_config = {
    "BAZEL_SNAP_OSS_BUCKET": "bazel-snapshot-hz",
    "BAZEL_SNAP_OSS_ENDPOINT_INTERNAL": "cn-hangzhou.oss-internal.aliyun-inc.com",
    "BAZEL_SNAP_OSS_ENDPOINT_EXTERNAL": "oss-cn-hangzhou.aliyuncs.com",
    "BAZEL_SNAP_OSS_ID": None,
    "BAZEL_SNAP_OSS_KEY": None,
    "BAZEL_SNAP_DBG_LOG": None,
}
log_lock_ = "/tmp/bazel_snap_client.lock"

# This block is only used in update mode
try:
    import oss2
    import requests

    class OssBucket:
        def __init__(self, endpoint, id, key, bucket, timeout=None):
            # disable debug messages from oss module
            logging.getLogger("oss2").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)

            auth = oss2.Auth(id, key)
            if not endpoint.startswith("http://"):
                endpoint = "http://" + endpoint
            self.oss_endpoint = endpoint
            self.oss_bucket = bucket
            self.bucket = oss2.Bucket(auth, endpoint, bucket, connect_timeout=timeout)

        def exists(self, path):
            # object_exists() will print a exception message at level INFO
            # this message is useless. set the level to ERROR to skip it.
            lvl = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)
            flag = self.bucket.object_exists(path)
            # restore log level
            logging.getLogger().setLevel(lvl)
            return flag

        def upload_file(self, path, local_path):
            return self.bucket.put_object_from_file(path, local_path)


except Exception:
    pass


def oss_eps():
    eps = [env_config["BAZEL_SNAP_OSS_ENDPOINT_EXTERNAL"]]
    if not running_on_ci():
        eps.append(env_config["BAZEL_SNAP_OSS_ENDPOINT_INTERNAL"])
    return eps


def gen_file_urls(path):
    urls = []
    # put lower priority in the front. bazel will sort reversely.
    for ep in oss_eps():
        urls.append(f"http://{env_config['BAZEL_SNAP_OSS_BUCKET']}.{ep}/{path}")
    return urls


def gen_oss_path(url, checksum_value, checksum_type):
    filename = os.path.basename(url)
    url = os.path.dirname(url)
    regex = r"\W"
    return f"cache/{re.sub(regex, '_', url)}/{checksum_type}_{checksum_value}/{filename}"


def check(args):
    urls = gen_file_urls(
        gen_oss_path(args.url[0], args.checksum_value, args.checksum_type)
    )
    for url in urls:
        print(f"FOUND_SNAPSHOT:{url}")


def update(args):
    path = gen_oss_path(args.url[0], args.checksum_value, args.checksum_type)
    uploaded_or_existed = False
    err_msg = ""
    for ep in oss_eps():
        err_msg += f"\nTry endpoint {ep}"
        bucket = OssBucket(
            ep,
            env_config["BAZEL_SNAP_OSS_ID"],
            env_config["BAZEL_SNAP_OSS_KEY"],
            env_config["BAZEL_SNAP_OSS_BUCKET"],
            timeout=1,
        )
        try:
            if not bucket.exists(path):
                # create a new bucket without timeout
                bucket2 = OssBucket(
                    ep,
                    env_config["BAZEL_SNAP_OSS_ID"],
                    env_config["BAZEL_SNAP_OSS_KEY"],
                    env_config["BAZEL_SNAP_OSS_BUCKET"],
                )
                bucket2.upload_file(path, args.downloaded)
            uploaded_or_existed = True
            break
        except oss2.exceptions.RequestError as e:
            if type(e.exception) is requests.exceptions.ConnectTimeout:
                continue  # ignore timeout issue
            else:
                err_msg += f"\n{e}"  # unknown error
        except Exception as e:
            err_msg += f"\n{e}"  # unknown error
    if not uploaded_or_existed:
        if env_config["BAZEL_SNAP_DBG_LOG"] is not None:
            from filelock import FileLock
            with FileLock(log_lock_):
                with open(env_config["BAZEL_SNAP_DBG_LOG"], "a") as ofh:
                    ofh.write(f"Upload {args.url[0]} failed:\n{err_msg}")
        raise Exception("uploading failed")
    print("UPLOAD_DONE")


def main():
    parser = argparse.ArgumentParser(
        description=f"""
This is a simple implementation of a custom snapshot client for url redirecting called by bazel.
All snapshots are stored on a OSS bucket. OSS bucket must set public-read permission.
OSS doesn't support file lock. So don't update snapshots concurrently by multiple users.
Use internal oss endpoint first to save network traffic if possible.

Environment variables used by both check & update modes:
- BAZEL_SNAP_OSS_BUCKET: {env_config["BAZEL_SNAP_OSS_BUCKET"]} (default)
- BAZEL_SNAP_OSS_ENDPOINT_INTERNAL: {env_config["BAZEL_SNAP_OSS_ENDPOINT_INTERNAL"]} (default)
- BAZEL_SNAP_OSS_ENDPOINT_EXTERNAL: {env_config["BAZEL_SNAP_OSS_ENDPOINT_EXTERNAL"]} (default)

Environment variables used only in update modes:
- BAZEL_SNAP_OSS_ID: (no default value)
- BAZEL_SNAP_OSS_KEY: (no default value)
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-C",
        action="store_const",
        const="check",
        dest="mode",
        help="check if given url has a snapshot in server",
    )
    parser.add_argument(
        "-U",
        action="store_const",
        const="update",
        dest="mode",
        help="update snapshot of given url in server",
    )
    parser.add_argument("url", nargs=1, help="url to query or update")
    parser.add_argument(
        "--checksum_value",
        metavar="VALUE",
        required=True,
        help="checksum value",
    )
    parser.add_argument(
        "--checksum_type", metavar="TYPE", required=True, help="checksum type"
    )
    parser.add_argument(
        "--downloaded",
        metavar="LOCAL_FILE",
        help="local file downloaded by bazel. used to update snapshot.",
    )

    args = parser.parse_args()
    assert args.mode is not None, "must set one of -C/-U"

    # update envs
    for k in env_config.keys():
        if k in os.environ:
            env_config[k] = os.environ[k]

    if args.mode == "update":
        assert args.downloaded is not None, "must give local file path in update mode"
        assert (
            env_config["BAZEL_SNAP_OSS_ID"] is not None
        ), "must setenv BAZEL_SNAP_OSS_ID in update mode"
        assert (
            env_config["BAZEL_SNAP_OSS_KEY"] is not None
        ), "must setenv BAZEL_SNAP_OSS_KEY in update mode"

    if env_config["BAZEL_SNAP_DBG_LOG"] is not None:
        from filelock import FileLock

        with FileLock(log_lock_):
            with open(env_config["BAZEL_SNAP_DBG_LOG"], "a") as ofh:
                ofh.write(
                    f"{args.mode} {args.url}, checksum={args.checksum_value}, checktype={args.checksum_type}"
                )
                if args.mode == "update":
                    ofh.write(f", local_file={args.downloaded}")
                ofh.write("\n")

    if args.mode == "check":
        check(args)
    elif args.mode == "update":
        update(args)


if __name__ == "__main__":
    main()
