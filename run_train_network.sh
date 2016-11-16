#! /bin/bash

DATASHAPE=253
CROPSHAPE=224
NETWORK=inception-bn
CLASSES=$(find ./data/train -mindepth 1 -type d | wc -l)
echo found number of classes: $CLASSES
SAMPLES=$(wc -l < "data/fish_train.lst")
echo found number of training samples: $SAMPLES
BATCHSIZE=64
LOGFILE=${NETWORK}.log
LOADMODEL=checkpoints/Inception-BN
EPOCHS=10

python train_imagenet.py \
        --data-dir ./data \
        --train-dataset train.rec \
        --val-dataset val.rec \
        --network ${NETWORK} \
        --num-classes ${CLASSES} \
        --lr 0.001 \
        --lr-factor 0.5 \
        --lr-factor-epoch 15 \
        --gpus 0 \
        --num-examples ${SAMPLES} \
        --num-epochs ${EPOCHS} \
        --batch-size ${BATCHSIZE} \
        --data-shape ${CROPSHAPE} \
        --save-model-prefix checkpoints/${NETWORK} \
        --log-dir checkpoints \
        --log-file $LOGFILE \
        --model-prefix ${LOADMODEL} \
        --finetune True \
        --finetune-scale 10 \
        --load-epoch 1
