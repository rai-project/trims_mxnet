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
cd /build
wget https://github.com/grpc/grpc/archive/v1.9.1.tar.gz
tar -xf v1.9.1
cd v1.9.1
make install prefix=$UPR_PREFIX
make install-plugins prefix=$UPR_INSTALL_PREFIX
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

| Name                       | Description | Default Value |
| -------------------------- | ----------- | ------------- |
| UPR_CLIENT                 |             |               |
| UPR_BASE_DIR               |             |               |
| UPR_MODEL_NAME             |             |               |
| UPR_PROFILE_TARGET         |             | profile.json  |
| UPR_INITIALIZE_EAGER       |             | false         |
| UPR_INITIALIZE_EAGER_ASYNC |             | false         |
| UPR_INPUT_CHANNELS         |             | 3             |
| UPR_INPUT_WIDTH            |             | 224           |
| UPR_INPUT_HEIGHT           |             | 224           |
| UPR_INPUT_MEAN_R           |             | 0             |
| UPR_INPUT_MEAN_G           |             | 0             |
| UPR_INPUT_MEAN_B           |             | 0             |
| -------------------------- | ----------- | ------------- |
| UPRD_EVICTION_POLICY       |             | NONE          |
| UPRD_MEMORY_PERCENTAGE     |             | 0.8           |

## How it Works

## Modifications

## Other

### Todo

[ ] Add more information about the system within the trace
[ ] Simplify running some of the examples

### Trace Viewer

https://github.com/rai-project/viz/blob/master/js/src/components/Trace/index.js
