#!/bin/bash

DATE=`date '+%Y-%m-%d-%H-%M-%S'`
export UPR_CLIENT=1
export UPR_INITIALIZE_EAGER=true
export MXNET_CPU_PRIORITY_NTHREADS=1
export OMP_NUM_THREADS=1

# export GLOG_v=4
export MXNET_ENGINE_TYPE=NaiveEngine
export MXNET_GPU_WORKER_NTHREADS=2

declare -a models=(
    bvlc_alexnet_1.0
    bvlc_googlenet_1.0
    bvlc_reference_caffenet_1.0
    bvlc_reference_rcnn_ilsvrc13_1.0
    dpn68_1.0
    dpn92_1.0
    inception_bn_3.0
    inception_resnet_2.0
    inception_3.0
    inception_4.0
    inceptionbn_21k_1.0
    locationnet_1.0
    network_in_network_1.0
    o_resnet101_2.0
    o_resnet152_2.0
    o_vgg16_1.0
    o_vgg19_1.0
    resnet34_2.0
    resnet50_2.0
    resnet50_1.0
    resnet101_2.0
    resnet101_1.0
    resnet152_11k_1.0
    resnet152_1.0
    resnet152_2.0
    resnet200_2.0
    resnet269_2.0
    resnext26_32x4d_1.0
    resnext50_32x4d_1.0
    resnext50_1.0
    resnext101_32x4d_1.0
    resnext101_1.0
    squeezenet_1.0
    squeezenet_1.1
    vgg16_sod_1.0
    vgg16_sos_1.0
    vgg16_1.0
    vgg19_1.0
    xception_1.0
    wrn50_2.0
    )

for i in "${models[@]}"
do
    echo start to run model = $i
    export UPR_MODEL_NAME=$i
    # first time
    ./image-classification-predict `hostname`_0

    # second time
    ./image-classification-predict `hostname`_1

    # third time
    ./image-classification-predict `hostname`_2
done

exit

for i in "${models[@]}"
do
    echo start to run model = $i
    export UPR_MODEL_NAME=$i
    # first time
    nvprof -f --export-profile `hostname`-${DATE}_profile_0.timeline.nvprof ./image-classification-predict `hostname`_0
    nvprof -f --export-profile `hostname`-${DATE}_profile_1.timeline.nvprof ./image-classification-predict `hostname`_1
    nvprof -f --export-profile `hostname`-${DATE}_profile_2.timeline.nvprof ./image-classification-predict `hostname`_2
done

exit
# nvprof -f --track-memory-allocations on --print-api-trace --export-profile `hostname`_profile_0.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_0.timeline.nvprof ./image-classification-predict
#nvprof -f --cpu-profiling on --print-gpu-trace --print-api-trace --export-profile `hostname`_profile_1.timeline.nvprof ./image-classification-predict
nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_1.timeline.nvprof ./image-classification-predict
# nvprof -f --cpu-profiling on --demangling on --export-profile profiles/`hostname`-${DATE}_profile_1.analysis.nvprof ./image-classification-predict
#nvprof -f --export-profile profiles/`hostname`-${DATE}_profile_2.timeline.nvprof ./image-classification-predict
