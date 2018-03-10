#!/bin/bash

DATE=`date '+%Y-%m-%d-%H-%M-%S'`
export UPR_CLIENT=1
export MXNET_CPU_PRIORITY_NTHREADS=1
export OMP_NUM_THREADS=1

export GLOG_v=0
export GLOG_logtostderr=0
export UPR_ENABLED=false
export MXNET_ENGINE_TYPE=NaiveEngine
export MXNET_GPU_WORKER_NTHREADS=1
export UPR_INITIALIZE_EAGER=true
./image-classification-predict
./image-classification-predict

exit
# nvprof -f --track-memory-allocations on --print-api-trace --export-profile `hostname`_profile_0.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_0.timeline.nvprof ./image-classification-predict
#nvprof -f --cpu-profiling on --print-gpu-trace --print-api-trace --export-profile `hostname`_profile_1.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_1.timeline.nvprof ./image-classification-predict
# nvprof -f --cpu-profiling on --demangling on --export-profile profiles/`hostname`-${DATE}_profile_1.analysis.nvprof ./image-classification-predict
#nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_2.timeline.nvprof ./image-classification-predict

