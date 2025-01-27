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
    --maxframes 2\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATXX\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATXY\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATXZ\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATYY\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATYZ\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATZZ\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield BETAX\
    --maxframes 2\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield BETAY\
    --maxframes 2\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ALPHA\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield W\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTXX\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTXY\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTXZ\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTYY\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTYZ\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTZZ\
    --maxframes 2\
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
    --maxframes 2\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATXX\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 20\
    --extent_y 20

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATXY\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATXZ\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATYY\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATYZ\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ATZZ\
    --maxframes 2\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield BETAX\
    --maxframes 2\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield BETAY\
    --maxframes 2\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.5\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield ALPHA\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield W\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 20\
    --extent_y 20

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTXX\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 20\
    --extent_y 20

"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTXY\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTXZ\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTYY\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTYZ\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 20\
    --extent_y 20
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 10\
    --plotfield GTZZ\
    --maxframes 2\
    --plot_cmap_max 1.5\
    --plot_cmap_min 0\
    --extent_x 20\
    --extent_y 20
"
)


# Run the commands in parallel using GNU Parallel
parallel --ungroup --jobs 0 '{}' ::: "${commands[@]}"