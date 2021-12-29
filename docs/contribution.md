# Contribute Code to BladeDISC

You're welcome to contribute to BladeDISC!
Please sign the [CLA](https://cla-assistant.io/alibaba/BladeDISC)
before contributing to BladeDISC community.  This document
introduces how to prepare your local development environment and our workflow.

## Local Development Environment

Some software is required on your host:

- [Docker](https://docs.docker.com/get-docker/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Git](https://git-scm.com/)

It's recommended to use the official development Docker image to
build and test your code:

``` bash
docker run --rm -it --gpus all -v $PWD:/workspace bladedisc/bladedisc:latest-devel-cuda11.0 bash
```

you can also find more images on [install with docker](./install_with_docker.md#download-a-bladedisc-docker-image).

## Submit a Pull Request to BladeDISC

BladeDISC uses [git branching model](https://nvie.com/posts/a-successful-git-branching-model/),
the following steps guide the usual contribution.

1. Fork

    Most commonly, forks are used to either propose changes to someone
    else's project or to use someone else's project as a starting point
    for your idea.  So, please file Pull Requests from your forked
    repo. To make a fork, just head over to the Github page and click
    ["Fork" button](https://help.github.com/articles/fork-a-repo/).

1. Clone

    To make a copy of your repo on your host, please run:

    ``` bash
    git clone git@github.com:your-github-account/BladeDISC.git
    ```

    please go to [build from source](./build_from_source.md) to check how
    to build and run tests.

1. Create a Local Feature Branch

    For each feature or fixing a bug, it's recommended to
    create a new feature branch before coding:

    ``` bash
    git checkout -b new_feature_branch
    ```

1. Keep Pulling Upstream

    BladeDISC is growing fast, many features are merged into the official
    repo, to notice the conflicts early, please pull the official repo
    often and it's easier to fix the conflicts.

    ``` bash
    git remote add upstream https://github.com/alibaba/BladeDISC
    git pull upstream main
    ```

1. Git Commit

    BladeDISC uses the [pre-commit] toolkit to check the code
    the style for each commit, please install the toolkit before committing:

    ``` bash
    python -m pip install pre-commit
    pre-commit install
    ```

    Once installation, you will see something like the following when
    you run `git commit` command:

    ``` text
    ➜  BladeDISC git:(new_feature) ✗ pre-commit run -a
    copyright_checker........................................................Passed
    ```

1. Create a Pull Request

    Once finish the new feature or bugfix, you can push the local
    work into your forked repo:

    ``` bash
    git push origin new-feature-branch
    ```

    the push allows you to create a new pull request when requesting
    the [official repo](https://github.com/alibaba/BladeDISC), please
    follows [this article](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
    to create a pull request.

    If your pull request is fixing an exiting issue, please fill in the key
    word [fixes #issue-id](https://help.github.com/articles/closing-issues-using-keywords/)
    to close the issue when the pull request is merged.

    Please feel free to assign a reviewer and assign an asocial label on
    your pull request page.
