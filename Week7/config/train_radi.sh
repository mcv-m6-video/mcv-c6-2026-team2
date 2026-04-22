#!/bin/bash

LOG_PATH=data/soccernetball/save_data
MODELS=( "tdeed_x3d_s_radi2" "tdeed_x3d_s_radi4" )

for MODEL in ${MODELS[@]}
do
    echo Starting $MODEL
    mkdir -p $LOG_PATH/$MODEL
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m main_spotting --model $MODEL | tee $LOG_PATH/$MODEL/log.txt
done
