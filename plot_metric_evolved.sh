#!/bin/bash

# Get the simulation name from command line argument
if [ -z "$1" ]; then
    echo "Usage: $0 <simulation_name>"
    exit 1
fi
simname=$1


# Define an array of commands
commands=(
"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield STRK\
    --maxframes 1000\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATXX\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATXY\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATXZ\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATYY\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATYZ\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATZZ\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield BETAX\
    --maxframes 1000\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield BETAY\
    --maxframes 1000\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ALPHA\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield W\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield GTXX\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600

"


"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield GTYY\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield GTZZ\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600
    

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield STRK\
    --maxframes 1000\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 40\
    --extent_y 40
"


"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATXX\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 40\
    --extent_y 40

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATXY\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATXZ\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATYY\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATYZ\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ATZZ\
    --maxframes 1000\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield BETAX\
    --maxframes 1000\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield BETAY\
    --maxframes 1000\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield ALPHA\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield W\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 40\
    --extent_y 40

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield GTXX\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 40\
    --extent_y 40

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield GTXY\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield GTXZ\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield GTYY\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield GTYZ\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield GTZZ\
    --maxframes 1000\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 40\
    --extent_y 40
"
)


# Run the commands in parallel using GNU Parallel
parallel --ungroup --jobs 0 '{}' ::: "${commands[@]}"