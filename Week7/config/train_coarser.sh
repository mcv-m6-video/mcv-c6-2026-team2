#!/bin/bash

LOG_PATH=data/soccernetball/save_data
# MODELS=( "tdeed_rny2_coarse" "tdeed_rny8_coarse" "x3d_l_coarse" "x3d_l_gru_coarse" )
MODELS=( "x3d_l_coarse" "x3d_l_tight" )

for MODEL in ${MODELS[@]}
do
    echo Starting $MODEL
    mkdir -p $LOG_PATH/$MODEL
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m main_spotting --model $MODEL | tee $LOG_PATH/$MODEL/log.txt
done
