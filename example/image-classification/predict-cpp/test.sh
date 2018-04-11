#!/bin/bash

DATE=`date '+%Y-%m-%d-%H-%M-%S'`
export OMP_NUM_THREADS=1
export GLOG_v=0
export GLOG_logtostderr=1

# export MXNET_ENGINE_TYPE=NaiveEngine
# export MXNET_GPU_WORKER_NTHREADS=1
# export MXNET_CPU_PRIORITY_NTHREADS=1

export UPR_ENABLED=true
export UPR_CLIENT=1
export UPR_INITIALIZE_EAGER=true
# export UPR_ENABLE_MEMORY_PROFILE=true

#UPR_MODEL_NAME=inception_3.0 ./image-classification-predict
#UPR_MODEL_NAME=bvlc_googlenet_1.0 ./image-classification-predict
UPR_MODEL_NAME=vgg16_1.0 ./image-classification-predict
#UPR_MODEL_NAME=squeezenet_1.0 ./image-classification-predict
#UPR_MODEL_NAME=bvlc_alexnet_1.0 ./image-classification-predict
# UPR_MODEL_NAME=bvlc_alexnet_1.0 ./image-classification-predict&
# UPR_MODEL_NAME=bvlc_googlenet_1.0 ./image-classification-predict &
# UPR_MODEL_NAME=bvlc_googlenet_1.0 ./image-classification-predict &

exit
# nvprof -f --track-memory-allocations on --print-api-trace --export-profile `hostname`_profile_0.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_0.timeline.nvprof ./image-classification-predict
#nvprof -f --cpu-profiling on --print-gpu-trace --print-api-trace --export-profile `hostname`_profile_1.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_1.timeline.nvprof ./image-classification-predict
# nvprof -f --cpu-profiling on --demangling on --export-profile profiles/`hostname`-${DATE}_profile_1.analysis.nvprof ./image-classification-predict
#nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_2.timeline.nvprof ./image-classification-predict

