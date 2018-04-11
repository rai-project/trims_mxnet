# Special thanks to https://github.com/pertusa for the Makefile
CFLAGS=-std=c++14

# Added for openblas
# export OPENBLAS_ROOT=/usr/local/opt/openblas

# CFLAGS+= -I${OPENBLAS_ROOT}/include
# LDFLAGS=-L${OPENBLAS_ROOT}/lib -lopenblas

# Added for opencv
CFLAGS+=`pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv` -lglog

# Added for mxnet
export MXNET_ROOT=`pwd`/../../../../mxnet-mirror

CFLAGS+=-I$(MXNET_ROOT)/include -I$(MXNET_ROOT)/src
CFLAGS+=-I/usr/include/openblas
CFLAGS+=-I$(MXNET_ROOT)/nnvm/include -I$(MXNET_ROOT)/mshadow
CFLAGS+=-I$(MXNET_ROOT)/dlpack/include -I$(MXNET_ROOT)/dmlc-core/include
CFLAGS+=-I$(MXNET_ROOT)/3rdparty
CFLAGS+=-I/usr/local/cuda
CFLAGS+=-DMSHADOW_USE_CBLAS
CFLAGS+=-DMXNET_USE_PROFILER=1
CFLAGS+=-DUSE_SSE=0
CFLAGS+=-DMSHADOW_USE_SSE=0
CFLAGS+=-DFMT_HEADER_ONLY=1
CFLAGS+=-DUSE_OPENMP=0
CFLAGS+=-DUSE_CUDNN=1
CFLAGS+=-DMSHADOW_USE_CUDNN=1
CFLAGS+=-DNO_OPENMP=1
CFLAGS+=-DUSE_CUDA=1
CFLAGS+=-DUSE_GPERFTOOLS=1
CFLAGS+=-Xcompiler -fno-builtin-malloc,-fno-builtin-calloc,-fno-builtin-realloc,-fno-builtin-free 
CFLAGS+=-DMXNET_USE_CUDA=1
#CFLAGS+= -Xcompiler -finstrument-functions 
LDFLAGS+=$(MXNET_ROOT)/lib/libmxnet.so -Xcompiler -finstrument-functions 
LDFLAGS+=-ltcmalloc

image-classification-predict: image-classification-predict.o
	nvcc -O3 -g -o image-classification-predict image-classification-predict.o $(LDFLAGS)

image-classification-predict.o: image-classification-predict.cc
	echo "CFLAGS = " $(CFLAGS)
	nvcc -O3  -g -c image-classification-predict.cc $(CFLAGS)
	
clean: 
	rm -f image-classification-predict
	rm -f *.d *.o *nvprof

lint:
	python ../../../dmlc-core/scripts/lint.py mxnet "cpp" ./
