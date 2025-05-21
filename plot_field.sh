#!/bin/bash

# Get the simulation name from command line argument
if [ -z "$1" ]; then
    echo "Usage: $0 <simulation_name>"
    exit 1
fi
simname=$1

module load parallel

commands=(
    "
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 6500\
    --extent_y 6500
"
    "
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 13000\
    --extent_y 13000
"
    "
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 3260\
    --extent_y 3260
"
    
"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 800\
    --extent_y 800
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 400\
    --extent_y 400
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 198\
    --extent_y 198
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 800\
    --extent_y 800
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 400\
    --extent_y 400
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 198\
    --extent_y 198
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPI\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 40\
    --extent_y 40
"
    
"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 3260\
    --extent_y 3260
"
"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 800\
    --extent_y 800
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 400\
    --extent_y 400
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 198\
    --extent_y 198
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPHI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 800\
    --extent_y 800
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 400\
    --extent_y 400
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 198\
    --extent_y 198
"

"
python makeplotxy.py \
    --simname $simname\
    --skipevery 1\
    --plotfield SPI2\
    --maxframes 10000\
    --plot_cmap jet\
    --plot_cmap_max 0.1\
    --plot_cmap_min -0.1\
    --extent_x 40\
    --extent_y 40
"
)



# Run the commands in parallel using GNU Parallel
parallel --ungroup --jobs 0 '{}' ::: "${commands[@]}"