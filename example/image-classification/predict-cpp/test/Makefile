# Special thanks to https://github.com/pertusa for the Makefile
CFLAGS=-std=c++14

# Added for openblas
# export OPENBLAS_ROOT=/usr/local/opt/openblas

# CFLAGS+= -I${OPENBLAS_ROOT}/include
# LDFLAGS=-L${OPENBLAS_ROOT}/lib -lopenblas


# Added for mxnet
export MXNET_ROOT=`pwd`/../../../../../mxnet-mirror

CFLAGS+=-I$(MXNET_ROOT)/include -I$(MXNET_ROOT)/src
CFLAGS+=-I$(MXNET_ROOT)/nnvm/include -I$(MXNET_ROOT)/mshadow
CFLAGS+=-I$(MXNET_ROOT)/dlpack/include -I$(MXNET_ROOT)/dmlc-core/include
CFLAGS+=-I$(MXNET_ROOT)/3rdparty
CFLAGS+=-I/usr/local/cuda
CFLAGS+=-DMSHADOW_USE_CBLAS
LDFLAGS+=$(MXNET_ROOT)/lib/libmxnet.so

test: test.o
	nvcc -O0 -g -o test test.o $(LDFLAGS)

test.o: test.cc
	nvcc -O0 -g -c test.cc ${CFLAGS}
	
clean: 
	rm test
	rm -f *.d *.o

lint:
	python ../../../dmlc-core/scripts/lint.py mxnet "cpp" ./
