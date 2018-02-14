# Special thanks to https://github.com/pertusa for the Makefile
CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall

# Added for openblas
# export OPENBLAS_ROOT=/usr/local/opt/openblas

# CFLAGS+= -I${OPENBLAS_ROOT}/include
# LDFLAGS=-L${OPENBLAS_ROOT}/lib -lopenblas

# Added for opencv
CFLAGS+= `pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`

# Added for mxnet
export MXNET_ROOT=`pwd`/../../../../mxnet-mirror

CFLAGS+=-Wall -I$(MXNET_ROOT)/include -I$(MXNET_ROOT)/src
CFLAGS+=-I$(MXNET_ROOT)/nnvm/include -I$(MXNET_ROOT)/mshadow
CFLAGS+=-I$(MXNET_ROOT)/dlpack/include -I$(MXNET_ROOT)/dmlc-core/include
CFLAGS+=-DMSHADOW_USE_CBLAS
LDFLAGS+=$(MXNET_ROOT)/lib/libmxnet.so

image-classification-predict: image-classification-predict.o
	g++ -std=c++14 -O3 -o image-classification-predict image-classification-predict.o $(LDFLAGS)

image-classification-predict.o: image-classification-predict.cc
	g++ -std=c++14 -O3 -c image-classification-predict.cc ${CFLAGS}
	
clean: 
	rm image-classification-predict
	rm -f *.d *.o

lint:
	python ../../../dmlc-core/scripts/lint.py mxnet "cpp" ./
