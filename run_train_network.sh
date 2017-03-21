#! /bin/bash

DATASHAPE=337
CROPSHAPE=299
NETWORK=inception-v3
LOADMODEL=models/Inception-7

CLASSES=$(find ./data/train -mindepth 1 -type d | wc -l)
echo found number of classes: $CLASSES
SAMPLES=$(wc -l < "data/fish_train.lst")
echo found number of training samples: $SAMPLES
BATCHSIZE=16
PREFIX=$1
echo $PREFIX
LOGFILE=${PREFIX}.log
EPOCHS=200

python train_imagenet.py \
        --data-dir ./data \
        --train-dataset train_${DATASHAPE}.rec \
        --val-dataset val_${DATASHAPE}.rec \
        --network ${NETWORK} \
        --num-classes ${CLASSES} \
        --lr 0.001 \
        --lr-factor 0.5 \
        --lr-factor-epoch 15 \
        --gpus $2 \
        --num-examples ${SAMPLES} \
        --num-epochs ${EPOCHS} \
        --batch-size ${BATCHSIZE} \
        --data-shape ${CROPSHAPE} \
        --save-model-prefix checkpoints/${PREFIX} \
        --log-dir checkpoints \
        --log-file $LOGFILE \
        --model-prefix ${LOADMODEL} \
        --finetune True \
        --finetune-scale 10 \
        --load-epoch 1
