import os
import numpy as np
from datetime import datetime
# Generate log-spaced outR values from 20.0 to 600.0
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_id', type=int, default=0, help='Split ID (0-3)')
parser.add_argument('--simname', type=str, default="250528_BBH_r70_moreplots_restart", help='Name of simulation directory')
args = parser.parse_args()

simname = args.simname
all_outR_values = np.logspace(np.log10(20.0), np.log10(320.0), 160)
# Use interleaved assignment - take every 4th value starting at split_id
outR_values = all_outR_values[args.split_id::4]
print(outR_values)

for outR in outR_values:
    cmd = f"bash run_parallel.sh --simname {simname} --total_workers 256 --maxframes 10000 --outR {outR:.1f} --fix_metric_error > plotlog{args.split_id}.txt 2>&1"
    print(f"Running with outR = {outR:.1f} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("command is", cmd)
    os.system(cmd)