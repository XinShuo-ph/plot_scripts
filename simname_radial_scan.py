import os
import numpy as np
from datetime import datetime
# Generate log-spaced outR values from 20.0 to 600.0
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_id', type=int, default=0, help='Split ID (0-3)')
# parser.add_argument('--simname', type=str, default="250528_BBH_r70_moreplots_restart", help='Name of simulation directory')
args = parser.parse_args()

# simname = args.simname
simnames = [
    '250520_BBH_r70_v04',
    '250514_BBH_r70',
    '250520_BBH_r70_v06',
    '20250520_BBH_r70_v07_correctid',
    '250125_BBH_r70_mu08_hires_longtime',
    '250519_BBH_r70_mu005_box3200_lowres'
]
all_outR_values = np.logspace(np.log10(100.0), np.log10(320.0), 8)
# Use interleaved assignment - take every 4th value starting at split_id
outR_values = all_outR_values[args.split_id::4]
print(outR_values)

for simname in simnames:
    for outR in outR_values:
        cmd = f"bash run_parallel.sh --simname {simname} --total_workers 128 --maxframes 10000 --outR {outR:.1f} --fix_metric_error > plotlog{args.split_id}.txt 2>&1"
        print(f"Running with outR = {outR:.1f} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("command is", cmd)
        os.system(cmd)