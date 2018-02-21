#!/bin/sh
MXNET_GPU_WORKER_NTHREADS=1 MXNET_ENGINE_TYPE=NaiveEngine nvprof -f --export-profile `hostname`_profile_2.timeline.nvprof ./image-classification-predict
