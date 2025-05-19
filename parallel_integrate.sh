#!/bin/bash

# use cpu nodes

# Define the number of frames
maxframes=128

# Loop over each frame index
for frameidx in $(seq 0 $((maxframes - 1)))
do
    # Create a new screen session for each frame index
    screen -dmS frame_$frameidx bash -c "
        python test_integrate.py \
        --simname 241118_BBH_r70_restart_longtime \
        --skipevery 1 \
        --maxframes $maxframes \
        --frameidx $frameidx \
        --outR 550 \
        --excise_factor 1.5 \
        --allfields
    "
done