# TRIMS (MXNet with Persistent GPU Memory)

## Reference

Dakkak, A., Li, C., De Gonzalo, S. G., Xiong, J., & Hwu, W-M. W. (2019). TrIMS: Transparent and isolated model sharing for low latency deep learning inference in function-as-a-service. In E. Bertino, C. K. Chang, P. Chen, E. Damiani, M. Goul, & K. Oyama (Eds.), Proceedings - 2019 IEEE International Conference on Cloud Computing, CLOUD 2019 - Part of the 2019 IEEE World Congress on Services (pp. 372-382). [8814494] (IEEE International Conference on Cloud Computing, CLOUD; Vol. 2019-July). IEEE Computer Society. https://doi.org/10.1109/CLOUD.2019.00067

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
