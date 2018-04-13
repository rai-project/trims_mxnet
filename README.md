# UPR (MXNet with Persistent GPU Memory)

## Installation

### Requirements

Setup your target location (make sure that `LD_LIBRARY_PATH` and `PATH` has been updated to include this)

```
export UPR_INSTALL_PREFIX=$HOME/.usr
export UPR_BASE_DIR=$HOME/carml/data/mxnet
```

make sure that both directories exist

```
mkdir -p $UPR_INSTALL_PREFIX
mkdir -p $UPR_BASE_DIR
```

Remember to update your `PKG_CONFIG_PATH`

```
export PKG_CONFIG_PATH=$UPR_INSTALL_PREFIX/lib/pkgconfig/:$PKG_CONFIG_PATH
```

#### C-ARES

```
wget https://c-ares.haxx.se/download/c-ares-1.13.0.tar.gz
tar -xf c-ares-1.13.0.tar.gz
cd c-ares-1.13.0
./buildconf
./configure --prefix=$UPR_INSTALL_PREFIX --disable-dependency-tracking --disable-debug
make install
```

#### Protobuf

```
wget https://github.com/google/protobuf/archive/v3.5.1.tar.gz
tar -xf v3.5.1.tar.gz
cd protobuf-3.5.1
./autogen.sh
./configure --prefix=$UPR_INSTALL_PREFIX --disable-dependency-tracking --disable-debug --with-zlib
make
make install
```

#### Protobuf-c

```
wget https://github.com/protobuf-c/protobuf-c/releases/download/v1.3.0/protobuf-c-1.3.0.tar.gz
tar -xf protobuf-c-1.3.0.tar.gz
cd protobuf-c-1.3.0
./configure --prefix=$UPR_INSTALL_PREFIX --disable-dependency-tracking --disable-debug
make install
```

#### GRPC

```
wget https://github.com/grpc/grpc/archive/v1.9.1.tar.gz
tar -xf v1.9.1
cd v1.9.1
make install prefix=$UPR_INSTALL_PREFIX
make install-plugins prefix=$UPR_INSTALL_PREFIX
```

#### OpenCV (Optional)

```
wget https://github.com/opencv/opencv/archive/3.4.1.tar.gz
tar -xf 3.4.1.tar.gz
cd opencv-3.4.1/
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$UPR_INSTALL_PREFIX -DWITH_CUDA=OFF -DWITH_OPENCL=OFF ..
make
make install
```

#### From APT-GET

Other requirements can be installed using APT . The base requirements are listed in the (MXNet installation guide)[https://mxnet.apache.org/install/index.html].

### Regenerate GRPC Code

It is not recommended to regenerate the protofile, since it's dependent on the protobuf version.
The only time you need to regenerate it is after updating it.

```
cd src/c_api
make
```

### Build Server

The server is part of the MXNet build process.

## Downloading Models

## Running

### Server

### Client

## Profiling

## Environment Variables

| Name                               | Description                           | Default Value    |
| ---------------------------------- | ------------------------------------- | ---------------- |
| UPR_ENABLED                        |                                       | true             |
| UPR_PROFILE_IO                     | only makes sense if UPR_ENABLED=false | true             |
| UPR_RUN_ID                         |                                       | [undefined]      |
| git                                |                                       | build_git_sha    |
| UPR_GIT_BRANCH                     |                                       | build_git_branch |
| UPR_GIT_TIME                       |                                       | build_git_time   |
| UPR_CLIENT                         |                                       |                  |
| UPR_BASE_DIR                       |                                       |                  |
| UPR_MODEL_NAME                     |                                       |                  |
| UPR_PROFILE_TARGET                 |                                       | profile.json     |
| UPR_INITIALIZE_EAGER               |                                       | false            |
| UPR_INITIALIZE_EAGER_ASYNC         |                                       | false            |
| UPR_INPUT_CHANNELS                 |                                       | 3                |
| UPR_INPUT_WIDTH                    |                                       | 224              |
| UPR_INPUT_HEIGHT                   |                                       | 224              |
| UPR_INPUT_MEAN_R                   |                                       | 0                |
| UPR_INPUT_MEAN_G                   |                                       | 0                |
| UPR_INPUT_MEAN_B                   |                                       | 0                |
| UPR_ENABLE_MEMORY_PROFILE          |                                       | false            |
| UPR_ENABLE_CUDA_FREE               |                                       | false            |
| UPR_SHARING_GRANULARITY            |                                       | model            |
| --------------------------         | -----------                           | -------------    |
| UPRD_EVICTION_POLICY               |                                       | LRU              |
| UPRD_ESTIMATION_RATE               |                                       | 1.0              |
| UPRD_MEMORY_PERCENTAGE             |                                       | 0.8              |
| UPRD_PERSIST_CPU                   |                                       | true             |
| UPRD_PERSIST_ONLY_CPU              | only persist on cpu memory            | false            |
| UPRD_WRITE_PROFILE                 | write server profile file             | false            |
| UPRD_ESTIMATE_WITH_INTERNAL_MEMORY | use internal memory info for estimate | true             |

## How it Works

## Modifications

## Other

### Todo

[ ] Add more information about the system within the trace
[ ] Simplify running some of the examples

### Trace Viewer

https://github.com/rai-project/viz/blob/master/js/src/components/Trace/index.js
Apache MXNet (incubating) for Deep Learning
=====

[![Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/incubator-mxnet/job/master/badge/icon)](http://jenkins.mxnet-ci.amazon-ml.com/job/incubator-mxnet/job/master/)
[![Documentation Status](http://jenkins.mxnet-ci.amazon-ml.com/job/incubator-mxnet-build-site/badge/icon)](https://mxnet.incubator.apache.org/)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

![banner](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png)

Apache MXNet (incubating) is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to ***mix*** [symbolic and imperative programming](https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts)
to ***maximize*** efficiency and productivity.
At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.

MXNet is also more than a deep learning project. It is also a collection of
[blue prints and guidelines](https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts) for building
deep learning systems, and interesting insights of DL systems for hackers.

[![Join the chat at https://gitter.im/dmlc/mxnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/mxnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

What's New
----------
* [Version 1.1.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.1.0) - MXNet 1.1.0 Release.
* [Version 1.0.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.0.0) - MXNet 1.0.0 Release.
* [Version 0.12.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.12.1) - MXNet 0.12.1 Patch Release.
* [Version 0.12.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.12.0) - MXNet 0.12.0 Release.
* [Version 0.11.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.11.0) - MXNet 0.11.0 Release.
* [Apache Incubator](http://incubator.apache.org/projects/mxnet.html) - We are now an Apache Incubator project.
* [Version 0.10.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.10.0) - MXNet 0.10.0 Release.
* [Version 0.9.3 Release](./docs/architecture/release_note_0_9.md) - First 0.9 official release.
* [Version 0.9.1 Release (NNVM refactor)](./docs/architecture/release_note_0_9.md) - NNVM branch is merged into master now. An official release will be made soon.
* [Version 0.8.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.8.0)
* [Updated Image Classification with new Pre-trained Models](./example/image-classification)
* [Python Notebooks for How to Use MXNet](https://github.com/dmlc/mxnet-notebooks)
* [MKLDNN for Faster CPU Performance](./MKL_README.md)
* [MXNet Memory Monger, Training Deeper Nets with Sublinear Memory Cost](https://github.com/dmlc/mxnet-memonger)
* [Tutorial for NVidia GTC 2016](https://github.com/dmlc/mxnet-gtc-tutorial)
* [Embedding Torch layers and functions in MXNet](https://mxnet.incubator.apache.org/faq/torch.html)
* [MXNet.js: Javascript Package for Deep Learning in Browser (without server)
](https://github.com/dmlc/mxnet.js/)
* [Design Note: Design Efficient Deep Learning Data Loading Module](https://mxnet.incubator.apache.org/architecture/note_data_loading.html)
* [MXNet on Mobile Device](https://mxnet.incubator.apache.org/faq/smart_device.html)
* [Distributed Training](https://mxnet.incubator.apache.org/faq/multi_devices.html)
* [Guide to Creating New Operators (Layers)](https://mxnet.incubator.apache.org/faq/new_op.html)
* [Go binding for inference](https://github.com/songtianyi/go-mxnet-predictor)
* [Amalgamation and Go Binding for Predictors](https://github.com/jdeng/gomxnet/) - Outdated
* [Large Scale Image Classification](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification)

Contents
--------
* [Documentation](https://mxnet.incubator.apache.org/) and  [Tutorials](https://mxnet.incubator.apache.org/tutorials/)
* [Design Notes](https://mxnet.incubator.apache.org/architecture/index.html)
* [Code Examples](https://github.com/dmlc/mxnet/tree/master/example)
* [Installation](https://mxnet.incubator.apache.org/install/index.html)
* [Pretrained Models](https://github.com/dmlc/mxnet-model-gallery)
* [Contribute to MXNet](https://mxnet.incubator.apache.org/community/contribute.html)
* [Frequent Asked Questions](https://mxnet.incubator.apache.org/faq/faq.html)

Features
--------
* Design notes providing useful insights that can re-used by other DL projects
* Flexible configuration for arbitrary computation graph
* Mix and match imperative and symbolic programming to maximize flexibility and efficiency
* Lightweight, memory efficient and portable to smart devices
* Scales up to multi GPUs and distributed setting with auto parallelism
* Support for Python, R, Scala, C++ and Julia
* Cloud-friendly and directly compatible with S3, HDFS, and Azure

Ask Questions
-------------
* Please use [discuss.mxnet.io](https://discuss.mxnet.io/) for asking questions.
* Please use [mxnet/issues](https://github.com/dmlc/mxnet/issues) for reporting bugs.

License
-------
Licensed under an [Apache-2.0](https://github.com/dmlc/mxnet/blob/master/LICENSE) license.

Reference Paper
---------------

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao,
Bing Xu, Chiyuan Zhang, and Zheng Zhang.
[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://github.com/dmlc/web-data/raw/master/mxnet/paper/mxnet-learningsys.pdf).
In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015

History
-------
MXNet emerged from a collaboration by the authors of [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva), and [purine2](https://github.com/purine/purine2). The project reflects what we have learned from the past projects. MXNet combines aspects of each of these projects to achieve flexibility, speed, and memory efficiency.
