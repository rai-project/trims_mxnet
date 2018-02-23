#!/bin/sh
export OMP_NUM_THREADS=1
export GLOG_v=4
export UPR_CLIENT=1
export MXNET_CPU_PRIORITY_NTHREADS=1
export MXNET_GPU_WORKER_NTHREADS=1
nvprof -f --export-profile `hostname`_profile_0.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile `hostname`_profile_1.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile `hostname`_profile_2.timeline.nvprof ./image-classification-predict
