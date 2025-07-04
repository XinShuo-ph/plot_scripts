import os
import numpy as np
from datetime import datetime
# Generate log-spaced outR values from 20.0 to 600.0
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_id', type=int, default=0, help='Split ID (0-3)')
parser.add_argument('--simname', type=str, default="20250620_tune_damping_n256_1", help='Name of simulation directory')
args = parser.parse_args()

simname = args.simname
excise_factors = np.linspace(1.1, 2.0, 10)
print(excise_factors)

for excise_factor in excise_factors:
    cmd = f"bash run_parallel.sh --withQ --simname {simname} --total_workers 128 --maxframes 10000 --outR 100.0 --excise_factor {excise_factor:.1f} > plotlog{args.split_id}.txt 2>&1"
    print(f"Running with excise_factor = {excise_factor:.1f} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("command is", cmd)
    os.system(cmd)