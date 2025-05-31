## Usage

For visualization, run
```bash
bash everythingplot.sh <simulation_name>
```
and wait for the screens to finish

For integrating the diagnostic quantities, run
```bash
python test_integrate_2d.py --fix_metric_error --maxframes 1000 --simname 250528_BBH_r70_moreplots_restart --outR 320.0
python test_integrate_2d_surface.py --maxframes 1000 --simname  250528_BBH_r70_moreplots_restart --outR 320.0
```
Or, to run faster 
```bash
commands=(
    "python test_integrate_2d.py --fix_metric_error --maxframes 1000 --simname 250528_BBH_r70_moreplots_restart --outR 650.0"
    "python test_integrate_2d.py --fix_metric_error --maxframes 1000 --simname 250528_BBH_r70_moreplots_restart --outR 320.0"
    "python test_integrate_2d.py --fix_metric_error --maxframes 1000 --simname 250528_BBH_r70_moreplots_restart --outR 160.0"
    "python test_integrate_2d.py --fix_metric_error --maxframes 1000 --simname 250528_BBH_r70_moreplots_restart --outR 90.0"
    "python test_integrate_2d.py --fix_metric_error --maxframes 1000 --simname 250528_BBH_r70_moreplots_restart --outR 45.0"    
    "python test_integrate_2d_surface.py --maxframes 1000 --simname  250528_BBH_r70_moreplots_restart --outR 650.0"
    "python test_integrate_2d_surface.py --maxframes 1000 --simname  250528_BBH_r70_moreplots_restart --outR 320.0"
    "python test_integrate_2d_surface.py --maxframes 1000 --simname  250528_BBH_r70_moreplots_restart --outR 160.0"
    "python test_integrate_2d_surface.py --maxframes 1000 --simname  250528_BBH_r70_moreplots_restart --outR 90.0"
    "python test_integrate_2d_surface.py --maxframes 1000 --simname  250528_BBH_r70_moreplots_restart --outR 45.0"
)
parallel --ungroup --jobs 0 '{}' ::: "${commands[@]}"
```

To run parallel version and cross check with serial version outputs like `plotGReX/250514_BBH_r70_2d_integrals_outR320.0_excise1.5.npy`, run 
```bash
bash run_parallel.sh --total_workers 64 --maxframes 64 --outR 160.0 --fix_metric_error --psipow_volume 0 --psipow_surface 0 > plotlog.txt 2>&1
```

Should see outputs like

```
✓ Time values match
✓ Column 1 matches perfectly
✓ Column 2 matches perfectly
✓ Column 3 matches perfectly
✓ Column 4 matches perfectly
✓ Column 5 matches perfectly
✓ Column 6 matches perfectly
✓ Column 7 matches perfectly
✓ Column 8 matches perfectly
✓ Column 9 matches perfectly
✓ Column 10 matches perfectly
✓ Column 11 matches perfectly
✓ Column 12 matches perfectly
✓ Column 13 matches perfectly
```

To do the analysis over all radii, run (on compute nodes since it takes ~2 hours using 256 workers in parallel), e.g.
```bash
python radial_scan.py --simname 250528_BBH_r70_moreplots_restart --split_id [0-3]
```

To create heatmaps of flux values in the plane of r and t, run (on login nodes since it takes less than 1 min), e.g.
```bash
python create_heatmap.py   --simname 250528_BBH_r70_moreplots_restart   --output_dir heatmaps   --startR 20.0   --endR 320.0   --numR 160   --symmetric_cmap   --cmap jet   --vmax '0.02,0.02,0.2,0.08'   --r_ticks 15   --no_title   --radius_range 20,200 --use_orbital_units
```