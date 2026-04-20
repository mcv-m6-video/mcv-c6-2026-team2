#!/bin/bash

LOG_PATH=data/soccernetball/save_data
MODELS=( "tdeed_rny2_tight" "tdeed_rny8_tight" "x3d_l_tight" "x3d_l_gru_tight" )

for MODEL in ${MODELS[@]}
do
    echo Starting $MODEL
    mkdir -p $LOG_PATH/$MODEL
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m main_spotting --model $MODEL | tee $LOG_PATH/$MODEL/log.txt
done
