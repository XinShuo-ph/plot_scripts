#!/bin/bash

# Load the parallel module
module load parallel

# Define an array of commands
commands=(
"
python makeplotxz_rho.py \
    --simname 240927_test_BBH_zboost\
    --skipevery 1\
    --maxframes 10000\
    --plot_log 1\
    --plot_cmap jet\
    --plot_cmap_max 1\
    --plot_cmap_min 0.000001\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxz_rho.py \
    --simname 240927_test_BBH_zboost\
    --skipevery 1\
    --maxframes 10000\
    --plot_log 1\
    --plot_cmap jet\
    --plot_cmap_max 1\
    --plot_cmap_min 0.000001\
    --extent_x 800\
    --extent_y 800
"

"
python makeplotxz_rho.py \
    --simname 240927_test_BBH_zboost\
    --skipevery 1\
    --maxframes 10000\
    --plot_log 1\
    --plot_cmap jet\
    --plot_cmap_max 1\
    --plot_cmap_min 0.000001\
    --extent_x 400\
    --extent_y 400
"

"
python makeplotxz_rho.py \
    --simname 240927_test_BBH_zboost\
    --skipevery 1\
    --maxframes 10000\
    --plot_log 1\
    --plot_cmap jet\
    --plot_cmap_max 1\
    --plot_cmap_min 0.000001\
    --extent_x 200\
    --extent_y 200
"

"
python makeplotxz_rho.py \
    --simname 240927_test_BBH_zboost\
    --skipevery 1\
    --maxframes 10000\
    --plot_log 1\
    --plot_cmap jet\
    --plot_cmap_max 1\
    --plot_cmap_min 0.000001\
    --extent_x 50\
    --extent_y 50
")

# Run the commands in parallel using GNU Parallel
parallel --ungroup --jobs 0 '{}' ::: "${commands[@]}"