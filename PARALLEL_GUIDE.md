to cross check with serial version outputs like `plotGReX/250514_BBH_r70_2d_integrals_outR320.0_excise1.5.npy`, run 
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