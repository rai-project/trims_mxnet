#!/bin/sh

DATE=`date '+%Y-%m-%d-%H-%M-%S'`
export UPR_CLIENT=1
export MXNET_CPU_PRIORITY_NTHREADS=1
export OMP_NUM_THREADS=1

# export GLOG_v=4
export MXNET_ENGINE_TYPE=NaiveEngine
export MXNET_GPU_WORKER_NTHREADS=2


export UPR_MODEL_NAME=alexnet
# first time
./image-classification-predict `hostname`_0

# second time
./image-classification-predict `hostname`_1

# third time
./image-classification-predict `hostname`_2

export UPR_MODEL_NAME=vgg16
# first time
./image-classification-predict `hostname`_0

# second time
./image-classification-predict `hostname`_1

# third time
./image-classification-predict `hostname`_2

export UPR_MODEL_NAME=resnet-152-11k
# first time
./image-classification-predict `hostname`_0

# second time
./image-classification-predict `hostname`_1

# third time
./image-classification-predict `hostname`_2

export UPR_MODEL_NAME=squeezenetv1
# first time
./image-classification-predict `hostname`_0

# second time
./image-classification-predict `hostname`_1

# third time
./image-classification-predict `hostname`_2

export UPR_MODEL_NAME=squeezenetv1.1
# first time
./image-classification-predict `hostname`_0

# second time
./image-classification-predict `hostname`_1

# third time
./image-classification-predict `hostname`_2

exit
# nvprof -f --track-memory-allocations on --print-api-trace --export-profile `hostname`_profile_0.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_0.timeline.nvprof ./image-classification-predict
#nvprof -f --cpu-profiling on --print-gpu-trace --print-api-trace --export-profile `hostname`_profile_1.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_1.timeline.nvprof ./image-classification-predict
# nvprof -f --cpu-profiling on --demangling on --export-profile profiles/`hostname`-${DATE}_profile_1.analysis.nvprof ./image-classification-predict
#nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_2.timeline.nvprof ./image-classification-predict