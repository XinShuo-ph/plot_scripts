#!/bin/bash

# Load the parallel module
module load parallel

# Define an array of commands
commands=(
    "
python makeplotxy_rho.py \
    --simname 240925_subcycling_BBH_2D_homoid_lowres128\
    --skipevery 1\
    --maxframes 10000\
    --plot_log 1\
    --plot_cmap jet\
    --plot_cmap_max 1\
    --plot_cmap_min 0.000001\
    --mu2 0.04\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy_rho.py \
    --simname 240925_subcycling_BBH_2D_homoid_lowres128\
    --skipevery 1\
    --maxframes 10000\
    --plot_log 1\
    --plot_cmap jet\
    --plot_cmap_max 1\
    --plot_cmap_min 0.000001\
    --mu2 0.04\
    --extent_x 400\
    --extent_y 400
"

"
python makeplotxy_rho.py \
    --simname 240925_subcycling_BBH_2D_homoid_lowres128\
    --skipevery 1\
    --maxframes 10000\
    --plot_log 1\
    --plot_cmap jet\
    --plot_cmap_max 1\
    --plot_cmap_min 0.000001\
    --mu2 0.04\
    --extent_x 198\
    --extent_y 198
"

"
python makeplotxy_rho.py \
    --simname 240925_subcycling_BBH_2D_homoid_lowres128\
    --skipevery 1\
    --maxframes 10000\
    --plot_log 1\
    --plot_cmap jet\
    --plot_cmap_max 1\
    --plot_cmap_min 0.000001\
    --mu2 0.04\
    --extent_x 50\
    --extent_y 50
")

# Run the commands in parallel using GNU Parallel
parallel --ungroup --jobs 0 '{}' ::: "${commands[@]}"