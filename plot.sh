

module load parallel

commands=(
    "
python makeplotxy.py \
    --simname 240925_BBH_3D_spongezone_2\
    --skipevery 1\
    --plotfield SPHI\
    --plotallfield \
    --maxframes 1000\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.50\
    --extent_x 1600\
    --extent_y 1600
"

"
python makeplotxy.py \
    --simname 240925_BBH_3D_spongezone_2\
    --skipevery 1\
    --plotfield SPHI\
    --plotallfield \
    --maxframes 1000\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.50\
    --extent_x 400\
    --extent_y 400
"

"
python makeplotxy.py \
    --simname 240925_BBH_3D_spongezone_2\
    --skipevery 1\
    --plotfield SPHI\
    --plotallfield \
    --maxframes 1000\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.50\
    --extent_x 198\
    --extent_y 198
"

"
python makeplotxy.py \
    --simname 240925_BBH_3D_spongezone_2\
    --skipevery 1\
    --plotfield SPI\
    --maxframes 1000\
    --plot_cmap_max 0.025\
    --plot_cmap_min -0.025\
    --extent_x 198\
    --extent_y 198
"

"
python makeplotxy.py \
    --simname 240925_BBH_3D_spongezone_2\
    --skipevery 1\
    --plotfield SPHI\
    --plotallfield \
    --maxframes 1000\
    --plot_cmap_max 0.5\
    --plot_cmap_min -0.50\
    --extent_x 40\
    --extent_y 40
"

"
python makeplotxy.py \
    --simname 240925_BBH_3D_spongezone_2\
    --skipevery 1\
    --plotfield SPI\
    --maxframes 1000\
    --plot_cmap_max 0.025\
    --plot_cmap_min -0.025\
    --extent_x 40\
    --extent_y 40
")



# Run the commands in parallel using GNU Parallel
parallel --ungroup --jobs 0 '{}' ::: "${commands[@]}"