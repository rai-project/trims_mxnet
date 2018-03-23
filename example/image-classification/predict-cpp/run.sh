#!/bin/sh

DATE=`date '+%Y-%m-%d-%H-%M-%S'`
export MXNET_CPU_PRIORITY_NTHREADS=1

# export GLOG_v=4
export MXNET_ENGINE_TYPE=ThreadedEngine

OUTPUT=out

UPR_MODEL_NAME=bvlc_alexnet_1.0 UPR_INPUT_WIDTH=227 UPR_INPUT_HEIGHT=227 ./image-classification-predict > $OUTPUT
UPR_MODEL_NAME=bvlc_googlenet_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=bvlc_reference_caffenet_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=bvlc_reference_rcnn_ilsvrc13_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=inceptionbn_21k_2.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=inception_3.0 UPR_INPUT_WIDTH=299 UPR_INPUT_HEIGHT=299 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=inception_4.0 UPR_INPUT_WIDTH=299 UPR_INPUT_HEIGHT=299 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=resnext50_32x4d_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=resnet101_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=resnet101_2.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=resnet152_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=resnet152_2.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=resnet50_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=squeezenet_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=squeezenet_1.1 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=vgg16_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=vgg16_sod_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=vgg19_1.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
UPR_MODEL_NAME=wrn50_2.0 UPR_INPUT_WIDTH=224 UPR_INPUT_HEIGHT=224 ./image-classification-predict >> $OUTPUT
