#!/bin/sh
export OMP_NUM_THREADS=1
export UPR_CLIENT=1
export MXNET_CPU_PRIORITY_NTHREADS=1
export MXNET_GPU_WORKER_NTHREADS=1
# export MXNET_ENGINE_TYPE=NaiveEngine

# export GLOG_v=4
export UPR_CLIENT=1
export MXNET_CPU_PRIORITY_NTHREADS=1
export MXNET_GPU_WORKER_NTHREADS=1
# nvprof -f --track-memory-allocations on --print-api-trace --export-profile `hostname`_profile_0.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile `hostname`_profile_0.timeline.nvprof ./image-classification-predict
nvprof -f --cpu-profiling on --print-gpu-trace --print-api-trace --export-profile `hostname`_profile_1.timeline.nvprof ./image-classification-predict
nvprof -f --cpu-profiling on --analysis-metrics --export-profile `hostname`_profile_1.analysis.nvprof ./image-classification-predict
nvprof -f --export-profile `hostname`_profile_2.timeline.nvprof ./image-classification-predict
